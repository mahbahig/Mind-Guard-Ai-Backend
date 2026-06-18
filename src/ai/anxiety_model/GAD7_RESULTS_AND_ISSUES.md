# GAD-7 screening model — recorded results and open issues

This note captures one successful training run (sklearn backend, wave-level granularity) and the **limitations and problems** to be aware of for the thesis and for any deployment discussion.

---

## How this run was produced

- **Working directory:** the project folder that **contains** the `src/` package (e.g. `anxiety_model`). `cd` there first.
- **Command:** `python -m src.train_gad7 --linear-backend sklearn`  
  **Do not** run `python src/train_gad7.py` by file path: the script uses **package relative imports** (`from .feature_pipeline import …`), which require `-m src.train_gad7` so Python loads `src` as a package. Otherwise you get `ImportError: attempted relative import with no known parent package`.
- **Optional flags:** `--no-verify` skips writing `models/verification/` plots (faster); `--debug` increases log verbosity; `--granularity person` / `--person-label mean|max` for person-level training; `--feature-top-k N` caps features after RFECV (default **22**).
- **Data default:** `data/raw` under that same project folder (Baigutanova-style CSVs), unless `--root` or `ANXIETY_DATA_DIR` is set.
- **Artifacts written:** `models/gad7_bundle.joblib` (bundle **version 2**), `models/training_metrics.json`, `models/feature_manifest.json`, and `models/verification/` (plots + OOF CSVs/JSON when verification is enabled — default on unless `--no-verify`).

---

## Implementation changelog (what changed in the codebase)

This section records the **GAD-7 / MindGuard** pipeline updates so the thesis and repo stay aligned.

### Data and feature table ([`src/wave_dataset.py`](src/wave_dataset.py))

- Drop redundant **raw** segment columns before aggregation (e.g. keep `hf` / `pnn50` / `rmssd` / `steps`; drop near-duplicate columns per collinearity analysis).
- **Rich aggregates** per wave for key numerics: mean, std, median, min, max, IQR, CV, count, slope; lighter stats for other columns where appropriate.
- **Day vs night** (e.g. 8–19h vs rest) sub-aggregates and day−night **deltas** for selected HRV metrics.
- **Demographics** from `survey.csv` as `demo_*` features (plus derived BMI where applicable).
- **Compliance / coverage:** `n_segments`, wave **span in days** (naming may appear as `window_span_days` in the table).
- **Person-level option:** `build_person_level_table` / CLI `--granularity person` with `--person-label mean|max`; person binary can use `binary_any_wave_ge5` when present.

### Feature selection ([`src/feature_pipeline.py`](src/feature_pipeline.py))

- **Variance** pruning, then **correlation** pruning (|r| > 0.85 between features, keep stronger |corr| to continuous GAD-7).
- **RFECV** with `RandomForestClassifier` + **GroupKFold** (RFE-style); fallback to stability RF top‑k if RFECV fails.
- **Grouped RF hyperparameter** search (`RandomizedSearchCV` + GroupKFold) for the binary forest.
- **Binary:** OOF-tuned elastic net vs RF weight, **Platt** (1D logistic on raw ensemble prob), **Youden** threshold on calibrated OOF scores; regression ensemble weight tuned on OOF MAE.

### Training entrypoint ([`src/train_gad7.py`](src/train_gad7.py))

- Outer **GroupKFold** for CV metrics and OOF predictions; inner linear models use **integer `cv=3`** on `ElasticNetCV` / `LogisticRegressionCV` to avoid nested `groups` issues with sklearn.
- **Variant selection:** primary **ensemble binary OOF AUC**; tie-break **elastic net regression CV MAE** (not MAE-only selection).
- **Per-variant metrics JSON:** `binary_label_distribution` with `n_positive_gad7_ge_5`, `n_negative_gad7_lt_5`, `prevalence_positive` (same wave-level definition as training labels).
- **Logs:** one INFO line per variant with `n_pos`, `n_neg`, prevalence before training details.
- **Convergence / warnings:** `LogisticRegressionCV` **`max_iter=20000`** (SAGA); `ElasticNetCV` **`max_iter=12000`**; pass **`use_legacy_attributes=False`** when the installed sklearn supports it; narrow **`warnings.filterwarnings`** for the sklearn 1.10 `use_legacy_attributes` **FutureWarning** so training logs stay clean.

### Verification ([`src/train_verification.py`](src/train_verification.py))

- In addition to confusion matrices, ROC, regression scatters: **calibration / reliability** plot, **RF feature importance** bar chart, **per-subject** prediction scatter, **learning curve** (grouped where applicable).

### Inference and services

- **[`src/inference_gad7.py`](src/inference_gad7.py):** loads bundle v2; **ensemble** + **Platt** calibrated `probability_gad7_ge_5`; returns **`screening_probability_threshold`** (Youden), **`raw_ensemble_probability`**, `binary_cutoff_used` (score ≥5), `variant_used`.
- **[`src/app.py`](src/app.py):** `Gad7ScreeningResponse` / `response_model` on `POST /predict_gad7_screening` for OpenAPI-aligned fields.
- **Repo root [`correlation_engine.py`](../correlation_engine.py):** when calling `/correlate`, if `gad7_screening` includes **`screening_probability_threshold`**, the engine uses it for the probability-based anxiety flag; otherwise it falls back to the fixed default in `THRESHOLDS`.

### Written results (this document)

- **Data snapshot:** explicit **34 / 100 / 134** class split and why AUC / balanced accuracy are reported.
- **§2 / §5:** examiner-oriented **defenses** for OOF-based calibration/threshold tuning at N=49 and for **capping features at ~22** vs RFECV’s higher `optimal_n` in logs.

---

## Data snapshot (this run)

| Item | Value |
|------|--------|
| Granularity | Wave (one row per subject × wave) |
| Rows | 134 |
| Subjects | 49 |
| Binary label (GAD-7 ≥ 5 vs &lt; 5) | **34 positive**, **100 negative** (~**25.4%** positive); same definition for both variants |
| Features after pipeline | 22 per variant (from ~148 `hrv_only` / ~160 `hrv_sleep` raw columns) |
| Variants trained | `hrv_only`, `hrv_sleep` |

**Class imbalance:** waves are **majority negative** (~75%). Raw accuracy would be misleading if the model always predicted “low”; **ROC AUC** is threshold-free and **balanced accuracy** (with a tuned probability threshold) summarize performance under imbalance. After training, `training_metrics.json` includes `binary_label_distribution` so the exact counts are always tied to the run.

---

## Variant selection (bundle default)

| Criterion | `hrv_only` | `hrv_sleep` |
|-----------|------------|-------------|
| **Ensemble binary OOF AUC** (primary) | 0.700 | **0.741** |
| Elastic net regression CV MAE (tie-break) | 2.351 | **2.244** |

**Selected variant:** `hrv_sleep` (higher ensemble screening AUC; better ENET MAE as tie-break).

**Rule:** maximize `ensemble_binary_oof_auc`, then minimize `elasticnet_regression_cv` MAE.

---

## Binary screening (GAD-7 ≥ 5) — headline OOF metrics

Rough summary from verification logs (out-of-fold, subject-grouped folds). Labels: **34** positive (≥5), **100** negative (&lt;5) at wave level.

### `hrv_only`

| Model / head | OOF AUC | Balanced accuracy (at tuned op.) | Notes |
|--------------|---------|----------------------------------|--------|
| Elastic net | ~0.56 | ~0.51 | Weak ranker |
| RF | ~0.69 | ~0.51 | Good AUC; default 0.5 threshold still poor for decisions |
| Ensemble + Platt + Youden | **~0.70** | **~0.67** | Calibrated prob + tuned threshold improves *usable* accuracy |

### `hrv_sleep` (selected)

| Model / head | OOF AUC | Balanced accuracy (at tuned op.) | Notes |
|--------------|---------|----------------------------------|--------|
| Elastic net | ~0.56 | ~0.49 | Weak |
| RF | ~0.74 | ~0.58 | Stronger |
| Ensemble + Platt + Youden | **~0.74** | **~0.71** | Best overall tradeoff in this run |

**Screening probability thresholds (Youden on OOF calibrated scores):** stored in the bundle / `training_metrics.json` (e.g. ~0.47 `hrv_only`, ~0.51 `hrv_sleep` for this run). Inference returns `screening_probability_threshold` so downstream code (e.g. correlation engine) need not assume 0.5.

---

## Regression (continuous GAD-7) — headline OOF metrics

| Variant | Best OOF MAE (approx.) | Spearman ρ (ensemble reg, approx.) |
|---------|------------------------|--------------------------------------|
| `hrv_only` | ~2.31 | ~0.35 |
| `hrv_sleep` | ~2.24 | ~0.39 |

Interpretation: **modest** agreement with true scores; expect **shrinkage** toward the dataset mean (true range 0–14+ but predictions often narrower).

---

## RF hyperparameter search (do not confuse with generalization)

Training logs include **RandomizedSearchCV + GroupKFold** lines reporting **best_auc ~0.77–0.79** on the tuning objective. That is **in-sample / inner-CV tuning**, not an unbiased external test. For reporting generalization, prefer **subject-wise OOF** ensemble metrics above.

---

## Problems and limitations (current state)

### 1. Small sample and repeated measures

- **49 subjects**, **134** wave rows: many rows share the same person. Metrics must stay **subject-aware** (GroupKFold); naive random splits would inflate performance.
- Effect sizes and AUCs have **high variance**; another random seed or fold layout can move numbers noticeably.

### 2. Optimism from using the same OOF stream for several steps

On the same out-of-fold predictions we tune:

- binary ensemble weight (elastic net vs RF),
- Platt scaling,
- Youden threshold,

then report confusion matrix / balanced accuracy at that threshold. That is **methodologically convenient** but can **slightly overstate** how well a *fully locked* model would do on **new subjects** who were never part of any tuning. A stricter story uses a **held-out subject set** or nested protocol where threshold/calibration are fit only on inner train.

**Examiner defense (why not a strict separate calibration / threshold holdout):** with only **49 unique subjects**, reserving an entirely disjoint subset *only* for Platt scaling and Youden thresholding would leave **very few subjects** in one branch of the data; parameter estimates for the threshold and calibration would be **high-variance and unstable**. Using one **subject-grouped OOF** stream for these steps is therefore a **small-sample necessity**, not only convenience. The tradeoff is modest **optimism bias**, which we state explicitly. A stronger scientific claim would add an **external cohort** or a **locked test subject list** never used in any tuning.

### 3. No external validation cohort

All tuning and evaluation are on **one public dataset**. Claims should be framed as **within-cohort screening performance**, not clinical utility in the general population.

### 4. Regression remains weak

- MAE ~2.2–2.4 on a skewed 0–21 scale is usable for **research dashboards** but not for replacing the GAD-7 questionnaire.
- Spearman ρ ~0.35–0.39 indicates **limited rank correlation** with true severity.

### 5. RFECV “optimal_n” vs final feature count

Logs may show RFECV **optimal_n** in the 40s–60s while the pipeline still **caps** at **22** features (`--feature-top-k`). The cap is intentional (stability / thesis interpretability) but means the printed “optimal_n” is not the final feature set size.

**Examiner defense (why not use the data-driven 40–60 features):** the **effective sample size for generalization is the number of subjects (49)**, not the 134 wave rows, because waves from the same person are correlated. With **40–60** predictors you are in a **high-dimensional regime relative to independent subjects** (curse of dimensionality), even with regularization and GroupKFold: variance explodes and coefficients become hard to interpret. RFECV optimizes **inner cross-validated performance on this matrix**; it does **not** encode a subject-level complexity budget. Capping at **~22** matches the original plan’s “~15–20” range and is a deliberate **bias–variance and interpretability** choice for **N = 49**, not an arbitrary dismissal of RFECV.

### 6. Software warnings during training

**Status (as implemented in `train_gad7.py`):** SAGA **`max_iter`** was raised (**20000** for `LogisticRegressionCV`, **12000** for `ElasticNetCV`), optional **`use_legacy_attributes=False`** is passed when supported, and a **narrow filter** suppresses the sklearn 1.10 **`use_legacy_attributes` FutureWarning** from `sklearn.linear_model._logistic`. A full training run on Python 3.14 + current sklearn should produce **clean logs** (no repeated logistic FutureWarnings or SAGA ConvergenceWarnings) under the default wave-level pipeline.

If new sklearn versions emit different messages, adjust filters or `max_iter` only for those specific warnings — avoid blanket `ignore` of all warnings.

### 7. Environment / path confusion risk

- Artifacts in this repo are under **`anxiety_model/models/`** (or a sibling folder name depending on checkout).
- Inference defaults to **`models/gad7_bundle.joblib`** relative to the package root; set **`GAD7_BUNDLE_PATH`** if you run the API from another cwd or duplicate tree.

### 8. Person-level mode (if used)

- `--granularity person` yields **49 rows**; metrics can look much stronger but are **very high variance** and easier to overfit. Treat wave-level as the primary scientific setting unless the thesis explicitly justifies person-level aggregation.

---

## Suggested wording for the thesis

> On the Baigutanova wearable cohort (N=49, wave-level n=134, binary label **~25%** positive for GAD-7≥5), a subject-grouped evaluation with HRV+sleep features achieved out-of-fold ROC AUC ≈0.74 for binary screening using a random forest–based ensemble with post-hoc calibration and a data-driven probability threshold. Continuous score prediction remained modest (MAE≈2.2, Spearman ρ≈0.39). Results are **exploratory** and require **external validation** before any clinical or deployment claims.

---

## Files to inspect after a run

- `models/training_metrics.json` — full JSON: per-variant **`binary_label_distribution`**, CV blocks, **`ensemble_binary_oof_auc`**, **`youden_j`**, **`platt_coef` / `platt_intercept`**, **`ensemble_binary_weight_enet`**, **`screening_prob_threshold`**, **`selected_variant`** (top-level), etc.
- `models/gad7_bundle.joblib` — serialized pipelines + `binary_screening` block for inference.
- `models/feature_manifest.json` — feature column lists per variant, selected variant, screening threshold alias for services.
- `models/verification/*` — ROC, confusion matrices, calibration reliability, feature importance, learning curve, per-subject scatter, OOF prediction CSVs.

---

*MindGuard / graduation project — GAD-7 screening. Last expanded with implementation changelog, run instructions, correlation-engine integration, and training-script log/metrics updates. Refresh numbers in the tables if you change data, seeds, or `--feature-top-k`.*
