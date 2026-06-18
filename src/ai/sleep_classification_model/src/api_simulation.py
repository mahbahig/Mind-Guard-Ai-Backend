import os
import sys
import json
import random
import numpy as np
import joblib
from scipy.signal import medfilt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from model_training import predict
    from feature_engineering import build_feature_vector
    from preprocessing import preprocess_all_epochs
except ImportError as e:
    print(f"Warning: Could not import necessary modules: {e}")

app = FastAPI(
    title="MindGuard Sleep Classification API",
    description="Real-time sleep/wake classification from raw 30-second sensor epochs.",
    version="3.1.0",
)

SIMULATION_MODEL = None
SIMULATION_RFE = None
SIMULATION_SCALER = None
WAKE_THRESHOLD = 0.5
IS_READY = False
TEST_SUBJECTS = []

DEFAULT_DATA_DIR = (
    r"E:\sleepdata\dreamt-dataset-for-real-time-sleep-stage-estimation-"
    r"using-multisensor-wearable-technology-2.1.0\data_100Hz"
)
HELD_OUT_RANGE = (102, 103)


# ──────────────────────────────────────────────
#  Schemas
# ──────────────────────────────────────────────

class EpochInput(BaseModel):
    bvp: List[float]
    hr: List[float]
    ibi: List[float]
    eda: List[float]
    acc: List[List[float]]


class PredictionResponse(BaseModel):
    predicted_stage: int
    probabilities: List[float]
    confidence: float
    message: str


class EpochPrediction(BaseModel):
    epoch_index: int
    predicted_stage: int
    true_label: int
    confidence: float


class SleepBiomarkers(BaseModel):
    tst_minutes: float
    sol_minutes: float
    waso_minutes: float
    sleep_efficiency_pct: float


class NightSimulationResponse(BaseModel):
    subject_index: int
    n_epochs: int
    epoch_predictions: List[EpochPrediction]
    biomarkers: SleepBiomarkers
    accuracy: float
    summary: str


# ──────────────────────────────────────────────
#  Sleep biomarker computation
# ──────────────────────────────────────────────

def compute_sleep_biomarkers(predictions, epoch_duration_sec=30):
    preds = list(predictions)
    n = len(preds)
    if n == 0:
        return {"tst_minutes": 0, "sol_minutes": 0, "waso_minutes": 0, "sleep_efficiency_pct": 0}

    total_min = n * epoch_duration_sec / 60.0
    tst = sum(1 for p in preds if p == 1) * epoch_duration_sec / 60.0

    first_sleep = next((i for i, p in enumerate(preds) if p == 1), None)
    if first_sleep is None:
        return {"tst_minutes": 0.0, "sol_minutes": total_min, "waso_minutes": 0.0, "sleep_efficiency_pct": 0.0}

    sol = first_sleep * epoch_duration_sec / 60.0
    last_sleep = len(preds) - 1 - next(i for i, p in enumerate(reversed(preds)) if p == 1)
    after_onset = preds[first_sleep:last_sleep + 1]
    waso = sum(1 for p in after_onset if p == 0) * epoch_duration_sec / 60.0
    se = (tst / total_min * 100.0) if total_min > 0 else 0.0

    return {
        "tst_minutes": round(tst, 1),
        "sol_minutes": round(sol, 1),
        "waso_minutes": round(waso, 1),
        "sleep_efficiency_pct": round(se, 1),
    }


# ──────────────────────────────────────────────
#  Startup — load model + held-out test subjects
# ──────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global SIMULATION_MODEL, SIMULATION_RFE, SIMULATION_SCALER, WAKE_THRESHOLD
    global IS_READY, TEST_SUBJECTS

    print("Initializing Sleep Classification API...")

    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(proj_root, "results", "models", "master_rf.joblib")
    rfe_path = os.path.join(proj_root, "results", "models", "master_rfe.joblib")
    scaler_path = os.path.join(proj_root, "results", "models", "master_scaler.joblib")
    params_path = os.path.join(proj_root, "results", "models", "best_params.json")

    if not (os.path.exists(model_path) and os.path.exists(rfe_path)):
        print(f"Model not found at {model_path}")
        print("Run 'python src/train_master_model.py' first.")
        return

    try:
        print(f"Loading model from: {model_path}")
        SIMULATION_MODEL = joblib.load(model_path)
        SIMULATION_RFE = joblib.load(rfe_path)
        if os.path.exists(scaler_path):
            SIMULATION_SCALER = joblib.load(scaler_path)
            print("Scaler loaded.")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                bp = json.load(f)
            WAKE_THRESHOLD = float(bp.get("wake_threshold", 0.5))
            print(f"Wake threshold loaded: {WAKE_THRESHOLD}")
        IS_READY = True
        print("Model ready.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    data_dir = os.environ.get("DREAMT_DATA_DIR") or DEFAULT_DATA_DIR
    if not os.path.isdir(data_dir):
        print(f"Test data dir not found: {data_dir}")
        print("Held-out subjects will not be available for /simulate_night.")
        return

    try:
        from evaluate import set_eval_mode, _load_local_csv_data
        set_eval_mode("2-stage")
        print(f"Loading held-out subjects S{HELD_OUT_RANGE[0]}-S{HELD_OUT_RANGE[1]}...")
        raw = _load_local_csv_data(fs=100, data_dir=data_dir, sid_range=HELD_OUT_RANGE)
        if raw:
            for subj in raw:
                n_ep = len(subj["labels"])
                epochs = []
                labels = []
                for i in range(n_ep):
                    epochs.append({
                        "bvp": subj["bvp"][i].copy(),
                        "hr": subj["hr"][i].copy(),
                        "ibi": subj["ibi"][i].copy(),
                        "eda": subj["eda"][i].copy(),
                        "acc": subj["acc"][i].copy(),
                    })
                    labels.append(int(subj["labels"][i]))
                TEST_SUBJECTS.append({"epochs": epochs, "labels": labels})
            total_ep = sum(len(s["epochs"]) for s in TEST_SUBJECTS)
            print(f"Loaded {len(TEST_SUBJECTS)} held-out subjects ({total_ep} epochs).")
        else:
            print("No held-out subjects found.")
    except Exception as e:
        print(f"Could not load held-out subjects: {e}")
        import traceback
        traceback.print_exc()


def _predict_single_epoch(sample):
    """Run the full pipeline on one epoch dict and return (predicted_stage, probabilities).

    Uses the calibrated WAKE_THRESHOLD: predict Wake (0) only if
    P(Wake) >= threshold, otherwise predict Sleep (1).
    """
    fs = 100
    bvp_2d = sample["bvp"].reshape(1, -1)
    pp_epochs = preprocess_all_epochs(bvp_2d, fs=fs)
    subject_data = {
        "bvp": pp_epochs,
        "hr": sample["hr"].reshape(1, -1),
        "ibi": sample["ibi"].reshape(1, -1),
        "eda": sample["eda"].reshape(1, -1),
        "acc": sample["acc"].reshape(1, sample["acc"].shape[0], 3),
    }
    X_features = build_feature_vector(subject_data, fs=fs)
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
    if SIMULATION_SCALER is not None:
        X_features = SIMULATION_SCALER.transform(X_features)
    _, y_proba, _ = predict(SIMULATION_MODEL, SIMULATION_RFE, X_features)
    probs = y_proba[0]
    stage = 0 if probs[0] >= WAKE_THRESHOLD else 1
    return stage, probs.tolist()


def _predict_all_epochs(subj_epochs):
    """Batch feature extraction + prediction for a full night (matches training/eval).

    Must be used when all epochs are available so Delta and Context features
    (53% of the 176-dim vector) match ``build_feature_vector`` at train time.
    Returns (raw_stages, y_proba) with raw_stages[i] in {0, 1}.
    """
    fs = 100
    n = len(subj_epochs)
    if n == 0:
        return np.array([], dtype=int), np.zeros((0, 2))

    bvp = np.stack([e["bvp"] for e in subj_epochs], axis=0)
    hr = np.stack([e["hr"] for e in subj_epochs], axis=0)
    ibi = np.stack([e["ibi"] for e in subj_epochs], axis=0)
    eda = np.stack([e["eda"] for e in subj_epochs], axis=0)
    acc = np.stack([e["acc"] for e in subj_epochs], axis=0)

    pp = preprocess_all_epochs(bvp, fs=fs)
    subject_data = {"bvp": pp, "hr": hr, "ibi": ibi, "eda": eda, "acc": acc}
    X = build_feature_vector(subject_data, fs=fs)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if SIMULATION_SCALER is not None:
        X = SIMULATION_SCALER.transform(X)
    _, y_proba, _ = predict(SIMULATION_MODEL, SIMULATION_RFE, X)
    raw_stages = np.where(y_proba[:, 0] >= WAKE_THRESHOLD, 0, 1).astype(int)
    return raw_stages, y_proba


# ──────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_ready": IS_READY,
        "n_test_subjects": len(TEST_SUBJECTS),
    }


@app.get("/")
def root():
    return {
        "message": "MindGuard Sleep Classification API.",
        "endpoints": ["/predict_epoch", "/simulate_night", "/health"],
    }


@app.post("/predict_epoch", response_model=PredictionResponse)
def predict_epoch(data: EpochInput):
    """Classify a single 30s epoch from raw sensor data."""
    if not IS_READY or SIMULATION_MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    try:
        sample = {
            "bvp": np.array(data.bvp, dtype=np.float64),
            "hr": np.array(data.hr, dtype=np.float64),
            "ibi": np.array(data.ibi, dtype=np.float64),
            "eda": np.array(data.eda, dtype=np.float64),
            "acc": np.array(data.acc, dtype=np.float64),
        }
        predicted_stage, probabilities = _predict_single_epoch(sample)
        confidence = float(max(probabilities))
        stage_names = ["Wake", "Sleep"]
        pred_str = stage_names[predicted_stage] if 0 <= predicted_stage < 2 else "Unknown"
        return PredictionResponse(
            predicted_stage=predicted_stage,
            probabilities=probabilities,
            confidence=confidence,
            message=f"Classified: {pred_str}",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


@app.get("/simulate_night", response_model=NightSimulationResponse)
def simulate_night(subject_index: Optional[int] = None):
    """
    Run a full-night simulation on a held-out test subject (never seen during training).
    Returns per-epoch predictions with ground truth, biomarkers, and accuracy.
    """
    if not IS_READY or SIMULATION_MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    if not TEST_SUBJECTS:
        raise HTTPException(status_code=503, detail="No held-out test subjects loaded.")

    if subject_index is not None:
        if subject_index < 0 or subject_index >= len(TEST_SUBJECTS):
            raise HTTPException(
                status_code=400,
                detail=f"subject_index must be 0-{len(TEST_SUBJECTS)-1}",
            )
        s_idx = subject_index
    else:
        s_idx = random.randint(0, len(TEST_SUBJECTS) - 1)

    subj = TEST_SUBJECTS[s_idx]
    epochs = subj["epochs"]
    labels = subj["labels"]
    n_epochs = len(epochs)

    raw_stages, y_proba = _predict_all_epochs(epochs)
    raw_preds = raw_stages.tolist()
    confidences = [float(max(y_proba[i])) for i in range(n_epochs)]
    true_labels = [1 if labels[e_idx] > 0 else 0 for e_idx in range(n_epochs)]

    # Temporal smoothing: 7-epoch median filter (3.5 min window)
    smoothed = medfilt(np.array(raw_preds, dtype=np.float64), kernel_size=7).astype(int)
    pred_list = smoothed.tolist()

    epoch_preds = []
    correct = 0
    for e_idx in range(n_epochs):
        epoch_preds.append(EpochPrediction(
            epoch_index=e_idx,
            predicted_stage=pred_list[e_idx],
            true_label=true_labels[e_idx],
            confidence=confidences[e_idx],
        ))
        if pred_list[e_idx] == true_labels[e_idx]:
            correct += 1

    accuracy = correct / n_epochs if n_epochs > 0 else 0.0
    bio = compute_sleep_biomarkers(pred_list)

    gt_wake = sum(1 for l in labels if l == 0)
    gt_sleep = n_epochs - gt_wake
    pred_wake = sum(1 for p in pred_list if p == 0)
    pred_sleep = n_epochs - pred_wake

    summary = (
        f"Held-out subject {s_idx} (S{HELD_OUT_RANGE[0] + s_idx}): "
        f"{n_epochs} epochs. "
        f"GT: {gt_wake} Wake / {gt_sleep} Sleep. "
        f"Pred: {pred_wake} Wake / {pred_sleep} Sleep. "
        f"Accuracy: {accuracy:.1%}. "
        f"TST={bio['tst_minutes']}min, SOL={bio['sol_minutes']}min, "
        f"WASO={bio['waso_minutes']}min, SE={bio['sleep_efficiency_pct']}%."
    )

    return NightSimulationResponse(
        subject_index=s_idx,
        n_epochs=n_epochs,
        epoch_predictions=epoch_preds,
        biomarkers=SleepBiomarkers(**bio),
        accuracy=round(accuracy, 4),
        summary=summary,
    )
