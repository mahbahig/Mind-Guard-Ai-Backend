import numpy as np
import neurokit2 as nk
import time

from src.preprocess import apply_butterworth_filter, normalize_signal, extract_hrv_features

print("Generating simulated data...")
sampling_rate = 256
duration = 30
ppg_signal = nk.ppg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=90)
noise = np.random.normal(0, 0.1, len(ppg_signal))
ppg_with_noise = ppg_signal + noise

print("Testing explicit extraction...")
start = time.time()
try:
    peaks_info = nk.ppg_findpeaks(ppg_with_noise, sampling_rate=sampling_rate)
    peaks = peaks_info["PPG_Peaks"]
    rr_intervals = np.diff(peaks) / sampling_rate * 1000 
    
    print(f"Finding peaks took {time.time() - start:.2f}s")
    
    start2 = time.time()
    sampen, _ = nk.entropy_sample(rr_intervals)
    print(f"SampEn took {time.time() - start2:.2f}s")
    
    start3 = time.time()
    hfd, _ = nk.fractal_higuchi(rr_intervals)
    print(f"Higuchi took {time.time() - start3:.2f}s")
    
except Exception as e:
    print("Error:", e)
print(f"Time taken: {time.time() - start:.2f} seconds")
