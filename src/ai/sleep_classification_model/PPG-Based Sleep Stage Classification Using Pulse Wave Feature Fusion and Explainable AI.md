---
tags: [research, ppg, sleep_tracking, machine_learning, explainable_ai]
---
[[Resources MOC]] | [[Affective Computing]] | [[AI MOC]]

# PPG-Based Sleep Stage Classification Using Pulse Wave Feature Fusion and Explainable AI

> [!ABSTRACT]
> This paper proposes an explainable random forest model for multi-stage sleep classification using statistical, temporal, and nonlinear dynamical features extracted from photoplethysmogram (PPG) pulse waves, achieving competitive performance while interpreting the model's decisions using SHAP.


- **Authors**: Florentin Smarandache, Satyasri Akula, Saleh I. Alzahrani, Farrukh Arslan, Amir Ijaz
- **Year**: 2025
- **Topic**: `Resources/Affective Computing`

## 2. The Core Synthesis
- **The "One Sentence" Summary**: By fusing statistical and nonlinear dynamics features of PPG signals and applying an explainable Random Forest model (GSCV-RF), this study provides an accessible, noninvasive, and interpretable alternative to traditional EEG/PSG sleep staging.

**Key Arguments**:
- Traditional sleep monitoring (EEG/PSG) is costly and unsuitable for continuous home monitoring, whereas PPG via wearables is affordable and accessible.
- Most existing ML models for sleep staging function as "black boxes," limiting their clinical utility.
- Fusing statistical time-domain features with nonlinear dynamical features (e.g., Higuchi Fractal Dimension, Hjorth Complexity) captures both signal variability and intrinsic irregularity, enhancing classification power.
- Explainable AI (XAI) via SHapley Additive exPlanations (SHAP) bridges the gap in clinical trust by interpreting model decisions, revealing distinct feature drivers for different sleep stages.

**Methodology**:
- **Data Acquisition**: Analyzed 9,394 expert-labeled 30-second sleep epochs from overnight PSG/PPG recordings of 10 subjects.
- **Preprocessing**: Applied baseline wander removal, Savitzky-Golay FIR filtering, and z-score normalization on PPG signals.
- **Feature Pipeline**: Extracted statistical, temporal, and nonlinear features; selected the top 10 most crucial features using Recursive Feature Elimination (RFE).
- **Classification**: Deployed a Grid Search Cross-Validation Random Forest (GSCV-RF) to perform 2-stage (82.56% accuracy), 3-stage, and 4-stage predictions against sibling methods like SVM and KNN.

## 3. Vault Integration (Contextual Mapping)
**Related Notes**:
- Supports the core research goals outlined in [[graduation project]] and [[meeting el ta5arog]].
- Direct technical overlap with [[Photoplethysmography-based HRV analysis]] and [[WESAD]] for feature engineering.

**The "Missing Link"**:
- While the vault's existing `neural network topics` and deep learning papers often pursue accuracy at the cost of interpretability, this paper highlights that **Explainable AI (SHAP)** is the missing link required to translate technical PPG algorithms into accepted clinical decision-making tools.

**Action Items**:
- [ ] Implement/Test Recursive Feature Elimination (RFE) and SHAP explainability on PPG datasets for [[graduation project]].
- [ ] Investigate the extraction of Higuchi and Katz Fractal Dimensions for [[Photoplethysmography-based HRV analysis]].
