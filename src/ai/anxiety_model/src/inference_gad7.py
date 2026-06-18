"""
Load gad7_bundle.joblib and run primary regression + binary screening.

Uses ensemble (elastic net + RF) + Platt calibration + Youden threshold when
``binary_screening`` is present in the bundle (v2).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .feature_pipeline import platt_predict


def default_bundle_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "models" / "gad7_bundle.joblib"


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "feature_manifest.json"


def load_bundle(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else default_bundle_path()
    if not p.is_file():
        raise FileNotFoundError(f"GAD-7 bundle not found: {p}")
    return joblib.load(p)


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else default_manifest_path()
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def predict_from_features(
    bundle: dict[str, Any],
    features: dict[str, float] | pd.Series,
    *,
    variant: str | None = None,
) -> dict[str, Any]:
    """
    features: mapping from training column name -> value (must include all required keys).
    variant: 'hrv_only' | 'hrv_sleep' | None (use bundle['selected_variant'])
    """
    var = variant or bundle.get("selected_variant", "hrv_only")
    block = bundle.get(var) or bundle["hrv_only"]
    names: list[str] = list(block["feature_names"])
    reg_en = block["pipelines"]["elasticnet_regression"]
    clf_en = block["pipelines"]["elasticnet_binary"]
    reg_rf = block["pipelines"]["rf_regression"]
    clf_rf = block["pipelines"]["rf_binary"]
    scr: dict[str, Any] = block.get("binary_screening") or {}

    if isinstance(features, dict):
        row = {k: float(features.get(k, np.nan)) for k in names}
    else:
        row = {k: float(features.get(k, np.nan)) for k in names}
    X = pd.DataFrame([row])[names]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    s_en = float(reg_en.predict(X)[0])
    s_rf = float(reg_rf.predict(X)[0])
    w_reg = float(scr.get("ensemble_regression_weight_enet", 0.5))
    score = w_reg * s_en + (1.0 - w_reg) * s_rf
    score = max(0.0, min(21.0, score))

    p_en = float(clf_en.predict_proba(X)[0, 1])
    p_rf = float(clf_rf.predict_proba(X)[0, 1])
    w_bin = float(scr.get("ensemble_weight_enet", 0.5))
    raw = w_bin * p_en + (1.0 - w_bin) * p_rf
    coef = np.array(scr.get("platt_coef", [1.0]), dtype=float)
    icept = float(scr.get("platt_intercept", 0.0))
    p_cal = platt_predict(raw, coef, icept)
    thr_screen = float(scr.get("screening_prob_threshold", 0.5))
    cutoff = float(bundle.get("binary_cutoff_gad7", 5.0))

    return {
        "predicted_gad7": round(score, 2),
        "probability_gad7_ge_5": round(p_cal, 4),
        "raw_ensemble_probability": round(raw, 4),
        "binary_cutoff_used": cutoff,
        "screening_probability_threshold": thr_screen,
        "variant_used": var,
        "disclaimer": "Screening / wellness estimate only — not a clinical diagnosis.",
    }
