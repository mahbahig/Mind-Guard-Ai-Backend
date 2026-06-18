# 🎛️ Project Orchestrator: Sleep Stage Classification

## 🔄 Configuration
- **Active Dataset & Hyperparameters:** [[Switchboard]]

## tasks 
- [x] **Model Correction & Class Rebalancing**
  - Module: [[Model_Correction_Agent]] 
  - *Follows: [[PPG-Based Sleep Stage Classification Using Pulse Wave Feature Fusion and Explainable AI]]*
- [x] **Data Audit**
  - Module: [[Data_Audit]] 
  - *Follows: [[PPG-Based Sleep Stage Classification Using Pulse Wave Feature Fusion and Explainable AI]]*

## 🛤️ Pipeline View & State Tracking

- [x] **1. Data (Local CSV: S002, S003)** 
  - Module: [[Data_Ingestion]] 
  - *Follows: Local files at 64 Hz, resampled to 100 Hz*
- [ ] **2. Preprocessing**
  - Module: [[Preprocessing]] 
  - *Follows: [[PPG-Based Sleep Stage Classification Using Pulse Wave Feature Fusion and Explainable AI]]*
- [x] **3. Features (PPG Pulse Wave)**
  - Module: [[Feature_Engineering]] 
  - *Follows: [[Extracted PPG Features for Sleep Staging]]* (Added Frequency & Deltas)
- [x] **4. Feature Selection & Model Training**
  - Module: [[Model_Training]] 
  - *Follows: [[PPG-Based Sleep Stage Classification Using Pulse Wave Feature Fusion and Explainable AI]]* (RFE top-10 → GSCV-RF + SHAP TreeExplainer)
- [ ] **5. Model Evaluation & Visualization**
  - Module: [[Model_Evaluation]]
  - *Runs: `python src/evaluate.py` → saves 4 plots to `results/plots/`*

## 🧪 Testing State
- [ ] Verification: [[Feature_Verification]]
- [ ] **Simulation & Real-life Testing**
  - Module: [[FastAPI_Simulation]]
  - *Details: Simulate real-life model testing with synthetic sleep data via a FastAPI endpoint.*
