# MindGuard — GAD-7 screening (Baigutanova-style features)

Novel **elastic net + random forest** models trained on the Baigutanova et al. (2025) wearable HRV dataset (*Scientific Data*), with **MindGuard** integration via the correlation engine.

## Quick start

1. Unzip the Figshare bundle locally (do not commit large data).
2. Install deps (from this folder):

   ```bash
   pip install -r requirements.txt
   ```

3. Train and write artifacts to `models/` (bundle **v2**: ensemble + Platt + Youden threshold, RFECV feature selection):

   ```bash
   python -m src.train_gad7 --root "C:\path\to\unzipped\figshare"
   ```

   Or set `ANXIETY_DATA_DIR` to that path and run without `--root`.

   **Granularity (wave vs person):** `--granularity wave` (default, one row per subject×wave) or `--granularity person` (one row per subject; `--person-label mean|max` for the regression target; binary uses `binary_any_wave_ge5` when available).

   **Feature cap:** `--feature-top-k` (default **22**) caps columns after variance/correlation pruning and **RFECV + GroupKFold** (RFE-style) selection.

   **Debug / verification (recommended once per machine):**

   ```bash
   python -m src.train_gad7 --debug --linear-backend sklearn
   ```

   Writes `models/verification/` with **out-of-fold** CSVs, JSON (confusion matrix counts, classification report, ROC AUC), and PNGs: confusion matrix, ROC, regression scatter, **reliability / calibration curve**, **RF feature importance**, **per-subject scatter**, **learning curve**. Small-N subject-wise CV: metrics near **0.5 AUC** are expected and still sanity-check the pipeline.

   **GPU (CUDA) for linear models:** sklearn **ElasticNet / LogisticRegression do not use the GPU**. RandomForest in sklearn is also **CPU-only**. To train the **linear** elastic-net–style heads on **PyTorch + CUDA**, use:

   ```bash
   python -m src.train_gad7 --linear-backend torch --torch-device cuda --epochs 2500
   ```

   RF baselines still run on CPU. If CUDA is missing, training falls back to **PyTorch on CPU** with a warning.

4. Run the screening API (recommended port **8003**):

   ```bash
   uvicorn src.app:app --host 0.0.0.0 --port 8003
   ```

   - `POST /predict_gad7_screening` — body: `{ "features": { "<column>": <float>, ... }, "variant": null }`
   - `GET /health` — reports whether `gad7_bundle.joblib` loaded
   - Override bundle path: `GAD7_BUNDLE_PATH`

## Artifacts

| File | Purpose |
|------|---------|
| `models/gad7_bundle.joblib` | Selected variant (by **ensemble binary OOF AUC**), elastic net + RF pipelines, `binary_screening` (Platt coefs, Youden `screening_prob_threshold`, ensemble weights) |
| `models/feature_manifest.json` | Ordered feature list + variant for inference |
| `models/training_metrics.json` | GroupKFold metrics (regression + binary) |

## Production feature path (plan **S1**)

- **Training / reproducibility:** precomputed `sensor_hrv_filtered.csv` (HeartPy-style columns) aggregated per person × wave — see `ANXIETY_GAD7_BAIGUTANOVA_PLAN.md`.
- **Runtime alignment:** buffer app PPG → NeuroKit HRV windows → **5-minute** mean/std, then **rename** keys to match `feature_manifest.json` (see `src/aggregate_windows.py`).

## Gateway call order

1. Sleep model → biomarkers (`tst_minutes`, `sol_minutes`, …).
2. HRV stress model → `stress_readings`.
3. **This service** → `POST /predict_gad7_screening` with aggregated features.
4. Correlation engine → `POST /correlate` on **port 8002** (default in repo) with optional `gad7_screening` (aliases: `predicted_gad7` → score, `probability_gad7_ge_5` → prob, `binary_cutoff_used` for **GAD score ≥5**, and **`screening_probability_threshold`** for the calibrated probability operating point — engine uses this when present instead of a fixed default).

## Ethics

Outputs are **screening / wellness only**, not a diagnosis. **GAD-7** questionnaire text is copyrighted (Spitzer et al.) if shown in-app. Dataset: **CC BY 4.0** — cite Baigutanova et al. when reporting results.
