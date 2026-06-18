# Feature Engineering Module

**Follows Implementation Guide:** [[Extracted PPG Features for Sleep Staging]]

## 📝 Description
Extracts a comprehensive set of **Statistical**, **Temporal**, and **Nonlinear** features from standard 30s PPG epochs for Random Forest classification. The top 10 most crucial features are then selected downstream via RFE — see [[Model_Training]].

## 💻 Code (Copy-Pasteable)

```python
import numpy as np
import scipy.stats as stats

def calculate_statistical_features(epoch):
    """
    Computes time-domain statistical metrics for a PPG waveform array.
    """
    # Central Tendency and Dispersion
    me = np.mean(epoch)
    std = np.std(epoch)
    mad = np.mean(np.abs(epoch - me))
    mabd = np.median(np.abs(epoch - np.median(epoch))) # Median Absolute Deviation
    iqr = stats.iqr(epoch)
    rms = np.sqrt(np.mean(np.square(epoch)))
    
    # Shape and Extremes
    sf = rms / me if me != 0 else 0           # Shape Factor
    tme25 = stats.trim_mean(epoch, 0.25)      # 25% Trimmed Mean
    tme50 = stats.trim_mean(epoch, 0.50)      # 50% Trimmed Mean
    gme = stats.gmean(np.abs(epoch) + 1e-10)  # Geometric Mean
    
    max_val = np.max(epoch)
    min_val = np.min(epoch)
    
    # Distribution
    sk = stats.skew(epoch)                    # Skewness
    ku = stats.kurtosis(epoch)                # Kurtosis
    ncm_3 = stats.moment(epoch, moment=3)     # 3rd Central Moment
    
    # Curve Length
    acl = np.sum(np.abs(np.diff(epoch)))      # Average Curve Length

    return [mad, mabd, iqr, ncm_3, acl, sf, me, std, rms, tme25, tme50, gme, max_val, min_val, sk, ku]

def calculate_nonlinear_features(epoch):
    """
    Analyzes intrinsic dynamics and PPG signal irregularity.
    """
    # Poincare Features (Lag 1)
    x = epoch[:-1]
    y = epoch[1:]
    sd1 = np.sqrt(np.var(x - y) / 2)
    sd2 = np.sqrt(np.var(x + y) / 2)
    sd1rsd2 = sd1 / sd2 if sd2 != 0 else 0

    # Hjorth Parameters
    dy = np.diff(epoch)
    ddy = np.diff(dy)
    var_zero = np.var(epoch)
    var_first = np.var(dy)
    var_second = np.var(ddy)
    
    hjm = np.sqrt(var_first / var_zero) if var_zero != 0 else 0  # Mobility
    hjc = np.sqrt(var_second / var_first) / hjm if hjm != 0 else 0  # Complexity
    
    # Placeholder for more complex nonlinear extractions (KFD / HFD / CCM)
    hfd = 0.0 # Requires Higuchi Fractal Dimension algorithm
    kfd = 0.0 # Requires Katz Fractal Dimension algorithm
    ccm = 0.0 # Requires Phase Space Reconstruction
    
    return [sd1, sd2, sd1rsd2, hjm, hjc, hfd, kfd, ccm]

def calculate_temporal_features(epoch, fs):
    """
    Captures timing-related characteristics of the PPG waveform within a 30s epoch.
    """
    # Zero-Crossing Rate: how often signal crosses zero
    zero_crossings = np.where(np.diff(np.sign(epoch)))[0]
    zcr = len(zero_crossings) / len(epoch)

    # Peak Count: number of local maxima in the epoch
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(epoch)
    peak_count = len(peaks) / (len(epoch) / fs)  # peaks per second

    # Mean Inter-Peak Interval (IPI): average time between successive peaks
    if len(peaks) > 1:
        ipi = np.mean(np.diff(peaks)) / fs  # in seconds
    else:
        ipi = 0.0

    # Rise Time: mean time from trough to next peak
    troughs, _ = find_peaks(-epoch)
    rise_times = []
    for peak in peaks:
        preceding = troughs[troughs < peak]
        if len(preceding) > 0:
            rise_times.append((peak - preceding[-1]) / fs)
    mean_rise_time = np.mean(rise_times) if rise_times else 0.0

    return [zcr, peak_count, ipi, mean_rise_time]

def build_feature_vector(epochs, fs=125):
    """
    Main extraction loop. 
    Expects (num_epochs, samples_per_epoch).
    fs: sampling frequency of the PPG signal (default 125 Hz for SLPDB).
    """
    feature_matrix = []
    for epoch in epochs:
        stat_feats = calculate_statistical_features(epoch)
        temp_feats = calculate_temporal_features(epoch, fs)
        nl_feats = calculate_nonlinear_features(epoch)
        feature_matrix.append(stat_feats + temp_feats + nl_feats)
        
    return np.array(feature_matrix)
```

## 🔗 Cross-References
- **RFE Feature Selection (Top 10):** Implemented in [[Model_Training]] — `RFE(n_features_to_select=10)` operates on the full feature matrix output by `build_feature_vector()`.

## 🐛 Known Issues/Bugs
- **Fractal Dimension Algorithms (HFD/KFD) & CCM** are placeholders (return `0.0`). Require a library like `antropy` or `neurokit2`.
- **Negative Values**: `gmean` uses `np.abs(epoch) + 1e-10` as a workaround for z-score normalized signals.
- **`fs` Parameter**: `build_feature_vector()` defaults to `fs=125` (SLPDB). Must be updated via `Switchboard` if using MESA or another dataset with a different sampling rate.
