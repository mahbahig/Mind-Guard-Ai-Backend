import numpy as np
from scipy.signal import savgol_filter, detrend
from scipy.stats import zscore

def remove_baseline_wander(signal, fs):
    """
    Removes baseline wander using simple linear detrending.
    (Future optimization: use a high-pass filter with a cutoff ~0.5Hz).
    """
    return detrend(signal, type='linear')

def apply_savgol_filter(signal, window_length=15, polyorder=3):
    """
    Applies Savitzky-Golay FIR filtering for smoothing without distorting 
    the signal tendency (preserves morphological shape).
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

def preprocess_all_epochs(epochs, fs):
    """
    Applies the full preprocessing pipeline to an array of epochs.
    Expects epochs of shape (num_epochs, samples_per_epoch).
    """
    preprocessed_epochs = []
    for epoch in epochs:
        preprocessed_epochs.append(preprocess_ppg_epoch(epoch, fs))
    return np.array(preprocessed_epochs)
