---
tags: [features, ppg, signal_processing, machine_learning]
---
[[Resources MOC]] | [[Affective Computing]] | [[AI MOC]]

# Extracted PPG Pulse Wave Features for Sleep Stage Classification

This note outlines the specific statistical, temporal, and nonlinear dynamical features extracted from photoplethysmogram (PPG) signals to classify sleep stages, as detailed in [[PPG-Based Sleep Stage Classification Using Pulse Wave Feature Fusion and Explainable AI]].

## Statistical Time-Domain Features
These features capture the essential descriptive properties of the PPG waveform and measure signal variability:
- **Mean Absolute Deviation (MAD)**: Measures average deviation from the mean (overall variability).
- **Median Absolute Deviation (MABD)**: robust measure of variability indicating how values deviate from the median (less sensitive to outliers).
- **Interquartile Range (IQR)**: The difference between the 75th and 25th percentiles (signal dispersion).
- **Nth Central Moment (NCM)**: Captures higher-order moments of the signal distribution structure.
- **Average Curve Length (ACL)**: Computes cumulative absolute differences of values over time (waveform complexity).
- **Shape Factor (SF)**: Ratio of RMS to the mean (describes overall shape).
- **Mean Value (ME)**: Average amplitude.
- **Standard Deviation (STD)**: Dispersion of values around the mean.
- **Root Mean Square (RMS)**: Square root of the mean squared values (signal energy).
- **Trimmed Mean (TME25, TME50)**: Mean values computed after excluding the lower/upper 25% or 50% values (reduces outlier effect).
- **Geometric Mean (GME)**: Alternative central tendency measure for skewed distributions.
- **Maximum Value (MAX)**: Highest recorded amplitude (peak signal fluctuations).
- **Minimum Value (MIN)**: Lowest recorded amplitude.
- **Skewness (SK)**: Asymmetry of the distribution.
- **Kurtosis (KU)**: Sharpness or flatness compared to a normal distribution.

## Nonlinear Dynamical Features
These features analyze the intrinsic dynamics and irregularity of the signal, highly crucial for analyzing autonomic nervous system activity:
- **Poincare SD1 (PSD1)**: Short-term variability in a Poincare plot (fast fluctuations in blood volume).
- **Poincare SD2 (PSD2)**: Long-term variability (sustained fluctuations).
- **Ratio of SD1/SD2 (SD1RSD2)**: Balance between short and long-term variability.
- **Complex Correlation Measure (CCM)**: Degree of complexity and correlation within the signal.
- **Hjorth Mobility (HjM)**: Rate of change (smoothness and frequency characteristics).
- **Hjorth Complexity (HjC)**: Irregularity of the waveform (how signal patterns evolve).
- **Higuchi Fractal Dimension (HFD)**: Fractal complexity measure of self-similarity and irregularity.
- **Katz Fractal Dimension (KFD)**: Estimates structural complexity varying across sleep stages.

## Feature Selection & Explainability
- **Recursive Feature Elimination (RFE)** is used to narrow down these extracted features to the 10 most correlated features for computational efficiency.
- According to SHAP analysis, features like **Higuchi Fractal Dimension**, **Hjorth Complexity**, **Geometric Mean**, **Central Moment**, and **Shape Factor** proved to be the most influential in distinguishing accurate four-stage sleep patterns.
