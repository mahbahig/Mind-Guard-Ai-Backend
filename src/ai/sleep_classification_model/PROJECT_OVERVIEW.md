# MindGuard — AI Models Overview

**Project:** MindGuard Mental Health Monitoring  
**Focus:** Real-time stress quantification and sleep quality tracking via wrist-worn PPG sensors  
**Tech Stack:** Python · FastAPI · Scikit-Learn · NeuroKit2 · MongoDB (planned) · WearOS/Apple Watch (planned)

---

## Architecture Overview

MindGuard uses two independent AI models that share the same raw data source—a continuous PPG (Photoplethysmography) signal from a smartwatch. Each model processes 30-second windows of this signal and produces a classification. Their outputs are then correlated by a higher-level engine to generate clinically meaningful mental health insights.

```
Smartwatch (PPG Signal)
        │
        ├──► [HRV Stress Model]  ──► Stress Level (0: None, 1: Moderate, 2: Severe)
        │
        └──► [Sleep Stage Model] ──► Sleep Stage (0: Wake, 1: Sleep)
                                            │
                              [Correlation Engine]
                                            │
                              ┌─────────────────────────┐
                              │ WASO · SOL · SE · TST   │
                              │ + Daytime Stress Trend  │
                              └─────────────────────────┘
                                            │
                                  Doctor/Patient Dashboard
```

---

## Model 1: HRV Stress Model

**Location:** `Ai_Models/Hrv_Model/`  
**Status:** ✅ Fully Operational (Training + Real-Time Inference + API running)

### What It Does
Classifies mental stress level in real-time from a 30-second PPG window.  
**Output:** `0` = No Stress · `1` = Moderate Stress · `2` = Severe Stress

### Dataset
- **SWELL-KW** (pre-extracted HRV features from knowledge work stress experiments)
- Located at: `data/final/train.csv` and `data/final/test.csv`
- Label mapping: `no stress → 0`, `time pressure → 1`, `interruption → 2`

### ML Model
| Component | Detail |
|---|---|
| Algorithm | Decision Tree Classifier |
| Key Features (11) | `MEAN_RR, MEDIAN_RR, SDRR, HR, LF_NU, HF, HF_PCT, TP, LF_HF, sampen, higuci` |
| Model Artifact | `models/stress_model.pkl` |
| Training Command | `python -m src.train` |

### Signal Processing Pipeline
1. **Butterworth High-Pass Filter** (0.5 Hz cutoff) — removes baseline wander
2. **Z-Score Normalization** — standardizes amplitude per window
3. **Peak Detection** via NeuroKit2 — identifies heartbeat locations
4. **HRV Feature Extraction** — 11 time-domain, frequency-domain, and nonlinear features

### Running the Server
```bash
# Terminal 1 — Start the API
python -m src.app

# Terminal 2 — Run the smartwatch simulator
python scripts/simulate_watch.py
```
API available at: `http://localhost:8000`  
Endpoint: `POST /predict_stress` → accepts raw PPG JSON array, returns stress label + features

---

## Model 2: Sleep Stage Classification Model

**Location:** `Ai_Models/Sleep-Classification_Model/`  
**Status:** ⚠️ Trained architecture ready — model `.joblib` artifacts need to be generated

### What It Does
Classifies each 30-second PPG epoch as **Wake** or **Sleep** throughout the night.  
**Output:** `0` = Wake · `1` = Sleep

### Dataset
- PPG/BVP signals from clinical sleep recordings (local CSV format)
- Located at: `data/` folder (raw signal CSVs, not pre-extracted features)

### ML Model
| Component | Detail |
|---|---|
| Algorithm | Random Forest + RFE (Recursive Feature Elimination) + GSCV |
| Feature Selection | Top 10 features selected by RFE from 64 total features |
| Model Artifacts | `results/models/master_rf.joblib` + `results/models/master_rfe.joblib` |
| Training Command | `python src/train_master_model.py` |
| API Command | `python -m uvicorn src.api_simulation:app --reload` |

### Signal Processing Pipeline
1. **Savitzky-Golay FIR Filter** — smooths signal, removes baseline wander
2. **Resampling to 100 Hz** — normalizes all recordings to uniform rate
3. **Feature Extraction** (64 features across 4 categories):
   - *Statistical* (16): Mean, Std, RMS, Kurtosis, Skewness, IQR, etc.
   - *Temporal* (4): Zero-Crossing Rate, Peak Count, Inter-Peak Interval, Rise Time
   - *Nonlinear* (8): Higuchi FD, Katz FD, Hjorth Mobility & Complexity, Poincaré SD1/SD2
   - *Frequency* (4): VLF, LF, HF spectral power, LF/HF Ratio
   - *Delta Features* (32): First-order differences of all above (temporal context)

### Validated Accuracy (Published Research)
| Source | Method | Result |
|---|---|---|
| Oxford Academic / SLEEP Journal (2023) | AI PPG vs. PSG gold standard | 90% sensitivity, 89% specificity |
| ResearchGate | Wrist PPG + HRV overnight | 94.1% accuracy, κ=0.71 |
| PubMed/NIH | Random Forest on PPG | 85.22% 2-stage accuracy |

---

## Why 2-Stage Sleep is Sufficient (Research-Backed)

A binary Wake/Sleep classifier produces these four clinical sleep biomarkers that directly differentiate anxiety/stress patients from healthy controls:

| Biomarker | Clinical Significance |
|---|---|
| **WASO** (Wake After Sleep Onset) | GAD patients show significantly higher WASO vs. healthy controls *(NIH/PubMed)* |
| **SOL** (Sleep Onset Latency) | Directly measures pre-sleep cognitive arousal (anxiety). Reduced by CBT *(NIH/PubMed)* |
| **SE** (Sleep Efficiency) | Low HRV (our stress metric!) independently predicts low SE *(Frontiers in Neuroscience, 2021)* |
| **TST** (Total Sleep Time) | Mild restriction reduces next-day SDNN and raises cortisol *(NIH crossover study)* |

> See full research report: `sleep_anxiety_research_report.md`

---

## Planned Integration Roadmap

- [ ] Generate `.joblib` artifacts by running `python src/train_master_model.py` in the Sleep model
- [ ] Port `preprocessing.py` and `feature_engineering.py` into `Hrv_Model/src/`
- [ ] Implement `/predict_sleep` endpoint in `Hrv_Model/src/app.py`
- [ ] Build correlation engine to compute WASO, SOL, SE, TST from epoch stream
- [ ] Upgrade `simulate_watch.py` to hit both endpoints and display combined output
- [ ] Migrate unified API to cloud backend (post-hardware integration)
- [ ] Replace simulator with live WearOS / Apple Watch data stream

---

## Repository Structure

```
Ai_Models/
├── Hrv_Model/                        # ✅ Stress Model (Operational)
│   ├── src/
│   │   ├── app.py                   # FastAPI server
│   │   ├── preprocess.py            # Signal filtering + HRV extraction
│   │   ├── train.py                 # Offline training pipeline
│   │   ├── inference.py             # Real-time classification
│   │   └── data_loader.py           # SWELL-KW CSV loader
│   ├── scripts/
│   │   └── simulate_watch.py        # Smartwatch PPG simulator
│   ├── data/final/                  # train.csv / test.csv (SWELL-KW)
│   ├── models/stress_model.pkl      # Trained Decision Tree artifact
│   └── requirements.txt
│
└── Sleep-Classification_Model/       # ⚠️ Sleep Model (Needs Training Run)
    ├── src/
    │   ├── api_simulation.py        # Standalone FastAPI server
    │   ├── feature_engineering.py   # 64-feature extraction pipeline
    │   ├── preprocessing.py         # Savitzky-Golay filter + resampling
    │   ├── model_training.py        # RFE + GSCV Random Forest
    │   └── train_master_model.py    # Master training entry point
    ├── results/models/              # ⚠️ Empty — run training to populate
    ├── sleep_anxiety_research_report.md
    └── requirements.txt
```
