"""
Production path (plan S1): aggregate short PPG-derived HRV windows into 5-minute
statistics compatible with Baigutanova-style training columns.

MindGuard today: NeuroKit features per ~30s window in ``Hrv_Model/src/preprocess.py``
(``MEAN_RR``, ``SDRR``, ``LF_NU``, ``HF``, ``LF_HF``, ...).

This module does **not** reproduce HeartPy columns exactly. For deployment you should:
1. **Preferred:** call the anxiety API with features exported from the same pipeline used
   at training time (``sensor_hrv_filtered.csv`` column names: ``HR_mean``, ``sdnn_std``, ...),
   built offline or from a watch SDK that mirrors the dataset.
2. **S1 (long-term):** collect many consecutive NeuroKit dicts over 5 minutes, then compute
   ``mean`` / ``std`` for each scalar key and **rename** into the closest training names
   (e.g. map ``SDRR`` -> ``sdnn`` for the mean column) before calling ``predict_from_features``.

Example aggregation (pseudo-code)::

    buffers: list[dict] = []  # each dict = one 30s NeuroKit feature row
    # every 5 minutes:
    df = pd.DataFrame(buffers)
    agg = {}
    for col in df.select_dtypes(include=[float, int]).columns:
        agg[f\"{col.lower()}_mean\"] = df[col].mean()
        agg[f\"{col.lower()}_std\"] = df[col].std()
    # then map keys to training ``feature_manifest.json`` names.

See ``models/feature_manifest.json`` after training for the exact ordered list.
"""

from __future__ import annotations

from pathlib import Path
import json


def load_expected_feature_names(manifest_path: str | Path | None = None) -> list[str]:
    """Return feature column names for the selected inference variant."""
    root = Path(__file__).resolve().parents[1]
    p = Path(manifest_path) if manifest_path else root / "models" / "feature_manifest.json"
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as f:
        m = json.load(f)
    var = m.get("inference_use_variant", "hrv_only")
    key = "feature_names_hrv_sleep" if var == "hrv_sleep" else "feature_names_hrv_only"
    return list(m.get(key, []))
