# Technical Documentation: PPG-Based Sleep Stage Classification

## 1. Project Overview

The **PPG-Based Sleep Stage Classification Project** offers a robust, machine-learning-driven solution to classify human sleep stages purely from non-invasive Photoplethysmography (PPG) pulse-wave signals. 

Traditionally, clinical sleep stage classification relies on polysomnography (PSG) setups which require bulky EEG sensors. This project solves the problem of invasive, unscalable sleep tracking by deriving high-accuracy, temporally sensitive estimations of a user's sleep state (e.g., Awake vs. Asleep, or 5-stage architectural sleep) utilizing easily accessible wrist-worn or optical sensor data.

## 2. Core Architecture

The system is highly modularized, adopting a strict pipeline architecture that segregates data handling from algorithmic evaluation and final model deployment. The core architectural relationship is as follows:

1. **Data Ingestion (`data_audit.py` & Handlers):** Processes `.csv` output from wearable sensors, securely loading continuous PPG and BVP (Blood Volume Pulse) data.
2. **Preprocessing (`preprocessing.py`):** Operates on the raw signal arrays to handle artifact removal, bandpass filtering, and dynamic resampling to uniformly cast user records to a standardized target sampling rate (e.g., 100 Hz).
3. **Feature Engineering (`feature_engineering.py`):** Acts as the mathematical core of the repository. It ingests pristine signals and outputs a high-dimensional vector containing time-domain, frequency-domain, and complex non-linear metrics (e.g., Katz and Higuchi Fractal Dimensions, Entropy markers).
4. **Machine Learning & Tuning (`model_training.py`):** Exclusively handles Recursive Feature Elimination (RFE) and Grid Search Cross-Validation (GSCV) applied to an advanced Random Forest (RF) classifier.
5. **Inference & Simulation (`api_simulation.py`):** Contains the deployment topology, bridging the resulting `.joblib` model artifacts to an asynchronous FastAPI REST backend for live, localized simulation of real-time signal classification.

## 3. Functional Logic

To understand how the project functions, observe the execution flow of our core inferential unit—the **Simulation Endpoint**:

### Step-by-Step Execution: `/simulate`
When a client application calls the `/simulate` endpoint under standard constraints, the request triggers a tightly wound functional logic block:

1. **Epoch Loading:** The API pre-loads verified 30-second patient signals into a cached system state during the FastAPI `startup` event to ensure low-latency responses.
2. **Array Restructuring:** A selected 30-second physiological epoch (3,000 continuous samples at 100 Hz) is cast into a `numpy` vector and dynamically reshaped for scikit-learn compatibility: `(1, -1)`.
3. **Pipeline Forward Pass:** The raw signal is pushed through `preprocess_all_epochs` and `build_feature_vector`, which automatically extract the mathematically relevant feature set (e.g., Hjorth mobility, peak-to-peak variability).
4. **Anomaly Control:** High-frequency extraction boundaries occasionally yield indeterminate numerics; the logic aggressively casts constraints via `np.nan_to_num(..., nan=0.0)`.
5. **Prediction Generation:** The extracted features are dimensionally reduced by the heavily fitted RFE selector to strictly include the top $N$ predictors. The Random Forest generates discrete stage predictions alongside probabilistic confidence intervals.
6. **Delivery:** The model packages this output into a strict Pydantic `PredictionResponse` base model yielding the true evaluation stage and predicted phase to the user.

```python
# Example Internal Execution Workflow
epochs = raw_signal.reshape(1, -1)

# Preprocessing & Feature Extraction
pp_epochs = preprocess_all_epochs(epochs, fs=fs)
X_features = build_feature_vector(pp_epochs, fs=fs)
X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

# Dimensionality Reduction & Prediction
y_pred, y_proba, _ = predict(SIMULATION_MODEL, SIMULATION_RFE, X_features)
```

## 4. Technical Stack & Techniques

### Technology Stack
- **Python 3.x**: Core scripting framework.
- **Scikit-Learn (sklearn)**: Model pipelines, hyperparameter optimizations, classification layers.
- **Imbalanced-Learn (imblearn)**: Statistical techniques for class parity correction.
- **SHAP (SHapley Additive exPlanations)**: Advanced global Interpretability library.
- **SciPy & NumPy**: Signal processing, interpolations, and vectorized mathematical operations.
- **FastAPI & Uvicorn**: High-performance, highly concurrent ASGI web server configuration framework for simulations.

### Complex Techniques and Rationale
1. **Recursive Feature Elimination (RFE)**
   Extracting complex entropy bounds and peak-derived statistics leaves the algorithm heavily susceptible to the *curse of dimensionality*. We leveraged `sklearn.feature_selection.RFE` dynamically layered over a base decision tree to iteratively strip mathematically weak features, restricting the model strictly to top-performing predictors (e.g., Top 10) to bolster real-world generalizability.
   
2. **Grid Search Cross-Validation (GSCV)**
   The project rejects static hyperparameter assumptions, utilizing an exhaustive combinatorial grid search executed over a 5-fold cross-validation scheme. This actively balances maximum depth, leaf constraints, and split criterias to penalize model variance strictly before `joblib` artifacts are generated.
   
3. **SMOTE (Synthetic Minority Over-sampling Technique)**
   Clinical sleep datasets natively hold heavy class imbalances (where users are dominantly asleep). The pipeline corrects boundary margins aggressively prior to splitting using SMOTE's synthetically derived k-nearest neighbor augmentations to allow unbiased Random Forest entropy splits.

4. **Biomedical Signal Complexity Extraction**
   Rather than treating the pulse volume as basic time-domain inputs, the mathematical codebase deploys Non-Linear Algorithms specifically calculated using overlapping windows, deriving **Katz Fractal Dimension** and **Higuchi Fractal Dimension** to analyze signal chaos as a proxy for neural arousal shifts.

## 5. Installation & Usage

### 1. Environment Preparation
Ensure you have Python installed, then resolve the core dependency stack.
```bash
pip install numpy scipy scikit-learn imbalanced-learn shap fastapi uvicorn pydantic joblib pandas
```

### 2. Master Model Training
Due to the dependency on dynamic dataset scaling, you must train the machine-learning pipeline to generate your specific `.joblib` weights before simulating the API context. Run the master training script:
```bash
python src/train_master_model.py
```
> *This script will cache your optimized weights `master_rf.joblib` & `master_rfe.joblib` inside `results/models/`*.

### 3. Executing the Simulation API
Initialize the Uvicorn ASGI server to host the local mock simulation.
```bash
python -m uvicorn src.api_simulation:app --reload
```
Navigate to your localized Swagger UI Dashboard at **http://127.0.0.1:8000/docs**. From here, you may natively execute `GET /simulate` directly inside the browser payload interface to observe your persisted ML model actively classifying sampled sleep signals.
