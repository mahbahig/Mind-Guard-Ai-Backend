"""
FastAPI service for GAD-7 screening (Baigutanova-style aggregated features).

Port 8003 recommended. Run from Anxiety_Model:
  uvicorn src.app:app --host 0.0.0.0 --port 8003
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference_gad7 import default_bundle_path, load_bundle, predict_from_features

app = FastAPI(
    title="MindGuard GAD-7 Screening",
    description="Predicts continuous GAD-7–aligned score and P(GAD-7>=5) from aggregated HRV/sensor features.",
    version="1.0.0",
)

_bundle: dict[str, Any] | None = None


@app.on_event("startup")
def _startup() -> None:
    global _bundle
    p = os.environ.get("GAD7_BUNDLE_PATH", "").strip() or str(default_bundle_path())
    try:
        _bundle = load_bundle(p)
    except FileNotFoundError:
        _bundle = None


class Gad7FeaturesPayload(BaseModel):
    """Feature names must match training columns (e.g. HR_mean, sdnn_std, ...)."""

    features: dict[str, float] = Field(..., description="Column name -> value for one person-wave aggregate row")
    variant: Optional[str] = Field(None, description="hrv_only or hrv_sleep; default from bundle")


class Gad7ScreeningResponse(BaseModel):
    """Aligned with ``inference_gad7.predict_from_features`` (bundle v2+)."""

    predicted_gad7: float
    probability_gad7_ge_5: float
    raw_ensemble_probability: float
    binary_cutoff_used: float
    screening_probability_threshold: float
    variant_used: str
    disclaimer: str


@app.post("/predict_gad7_screening", response_model=Gad7ScreeningResponse)
def predict_gad7_screening(payload: Gad7FeaturesPayload):
    if _bundle is None:
        raise HTTPException(
            status_code=503,
            detail="GAD-7 bundle not loaded. Train with: python -m src.train_gad7 --root <data/raw>",
        )
    try:
        return predict_from_features(_bundle, payload.features, variant=payload.variant)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing or invalid feature key: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/health")
def health():
    return {
        "status": "ok" if _bundle is not None else "degraded",
        "gad7_bundle_loaded": _bundle is not None,
        "selected_variant": _bundle.get("selected_variant") if _bundle else None,
    }


@app.get("/")
def root():
    return {"service": "MindGuard GAD-7", "docs": "/docs", "health": "/health"}
