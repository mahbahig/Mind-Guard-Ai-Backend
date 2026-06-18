import numpy as np
import scipy.signal as signal
import neurokit2 as nk

def apply_butterworth_filter(ppg_signal: np.ndarray, sampling_rate: int = 256) -> np.ndarray:
    """
    Applies a high-pass Butterworth filter (0.5 Hz cutoff) to eliminate DC noise 
    and baseline wander from the raw continuous PPG signal.
    """
    cutoff = 0.5  # Hz
    nyq = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyq
    
    # 4th order highpass filter
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    filtered_signal = signal.filtfilt(b, a, ppg_signal)
    
    return filtered_signal

def normalize_signal(ppg_signal: np.ndarray) -> np.ndarray:
    """
    Normalizes the signal within the 30-second window using Z-score normalization.
    """
    mean_val = np.mean(ppg_signal)
    std_val = np.std(ppg_signal)
    
    if std_val == 0:
        return ppg_signal - mean_val
    return (ppg_signal - mean_val) / std_val

import warnings

def extract_hrv_features(ppg_window: np.ndarray, sampling_rate: int = 256) -> dict:
    """
    Preprocesses a 30-second window of raw continuous PPG data and extracts the 
    required HRV features for stress classification.
    """
    # 1. Filter
    filtered_ppg = apply_butterworth_filter(ppg_window, sampling_rate)
    
    # 2. Normalize
    normalized_ppg = normalize_signal(filtered_ppg)
    
    # 3. Peak Detection & Extraction
    try:
        # Find peaks
        peaks_info = nk.ppg_findpeaks(normalized_ppg, sampling_rate=sampling_rate)
        peaks = peaks_info["PPG_Peaks"]
        
        # Check if enough peaks are detected for HRV
        if len(peaks) < 3:
            raise ValueError("Insufficient peaks detected in 30s window for HRV extraction.")
        
        # We need exactly these features:
        # ['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'HR', 'LF_NU', 'HF', 'HF_PCT', 'TP', 'LF_HF', 'sampen', 'higuci']
        
        # Get Time-domain, Frequency-domain, and Nonlinear features
        # Suppress warnings because 30s is too short for some long-term nonlinear indices (like DFA alpha2)
        # which neurokit2 tries to calculate anyway, causing console spam.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_rate)
            hrv_freq = nk.hrv_frequency(peaks, sampling_rate=sampling_rate)
            hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=sampling_rate)
            
            # Extract RR intervals (differences between successive peaks in ms)
            rr_intervals = np.diff(peaks) / sampling_rate * 1000 
            
            # Higuchi Fractal Dimension (using RR intervals instead of raw signal to prevent O(N^2) hang)
            if len(rr_intervals) > 1:
                hfd, _ = nk.fractal_higuchi(rr_intervals)
            else:
                hfd = 0.0

        # Mapping to the exact SWELL-KW format used in training
        # SDRR is equivalent to SDNN
        features = {
            'MEAN_RR': float(hrv_time['HRV_MeanNN'].values[0]),
            'MEDIAN_RR': float(hrv_time['HRV_MedianNN'].values[0]),
            'SDRR': float(hrv_time['HRV_SDNN'].values[0]),
            'HR': float(60000.0 / hrv_time['HRV_MeanNN'].values[0]), # Beats per minute
            'LF_NU': float(hrv_freq['HRV_LFn'].values[0]),
            'HF': float(hrv_freq['HRV_HF'].values[0]),
            'HF_PCT': float(hrv_freq['HRV_HFn'].values[0] * 100), # Normalized HF often corresponds to pct
            'TP': float(hrv_freq['HRV_TP'].values[0]),
            'LF_HF': float(hrv_freq['HRV_LFHF'].values[0]),
            'sampen': float(hrv_nonlinear['HRV_SampEn'].values[0]),
            'higuci': float(hfd)
        }
        
        # Clean up NaNs and Infs for JSON APIs and ML model input compliance
        for k, v in features.items():
            if np.isnan(v) or np.isinf(v):
                features[k] = 0.0
                
    except Exception as e:
        print(f"Error extracting HRV features: {e}")
        # Default fallback values to prevent 400 errors but maintain feature keys
        keys = ['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'HR', 'LF_NU', 'HF', 'HF_PCT', 'TP', 'LF_HF', 'sampen', 'higuci']
        return {k: 0.0 for k in keys}
        
    return features
