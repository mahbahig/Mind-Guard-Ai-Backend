# Preprocessing Module

**Follows Implementation Guide:** [[PPG-Based Sleep Stage Classification Using Pulse Wave Feature Fusion and Explainable AI]]

## 📝 Description
Cleans the raw PPG signals using baseline wander removal, Savitzky-Golay FIR filtering, and z-score normalization, as described in the reference paper.

## 💻 Code (Copy-Pasteable)

```python
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import zscore

def remove_baseline_wander(signal, fs):
    """
    Removes baseline wander using a high-pass filter or detrending.
    Here, simple linear detrending is used as a placeholder.
    """
    from scipy.signal import detrend
    return detrend(signal, type='linear')

def apply_savgol_filter(signal, window_length=15, polyorder=3):
    """
    Applies Savitzky-Golay FIR filtering for smoothing without distorting the signal tendency.
    """
    return savgol_filter(signal, window_length=window_length, polyorder=polyorder)

def normalize_signal(signal):
    """
    Applies z-score normalization to standardize amplitude.
    """
    return zscore(signal)

def preprocess_ppg_epoch(epoch, fs):
    """
    Full preprocessing pipeline for a single 30s PPG epoch.
    """
    epoch_no_bw = remove_baseline_wander(epoch, fs)
    epoch_filtered = apply_savgol_filter(epoch_no_bw)
    epoch_normalized = normalize_signal(epoch_filtered)
    
    return epoch_normalized
```

## 🐛 Known Issues/Bugs
- `remove_baseline_wander` currently uses linear detrending. A high-pass filter (e.g., cutoff 0.5Hz) or polynomial fitting might be more effective for robust baseline wander removal in PPG.
- `window_length` for `savgol_filter` must be odd and depends on the sampling frequency (`fs`). Make sure to tune it according to `Switchboard`.
