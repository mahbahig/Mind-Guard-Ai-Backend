import numpy as np
import scipy.stats as stats
from scipy.signal import find_peaks, welch


def _band_power_trapz(y, x):
    """Integrate spectrum band; works on NumPy 1.x (trapz) and 2.x (trapezoid)."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _unique_successive(x):
    """Return values at indices where the series changes (strip hold-interpolation duplicates)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return np.array([])
    mask = np.concatenate([[True], x[1:] != x[:-1]])
    return x[mask]


# ─── IBI / HRV features ─────────────────────────────────────────────

def calculate_hrv_features(ibi_epoch):
    """
    HRV-style metrics from IBI (seconds) time series within one epoch.
    Deduplicates held/upsampled IBI values before statistics.
    Returns 6 features: mean_IBI, SDNN, RMSSD, pNN50, CVSD, mean_HR_from_IBI
    """
    ibi_s = _unique_successive(ibi_epoch)
    ibi_s = ibi_s[np.isfinite(ibi_s) & (ibi_s > 0.05) & (ibi_s < 3.0)]
    if ibi_s.size < 2:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    mean_ibi = float(np.mean(ibi_s))
    sdnn = float(np.std(ibi_s, ddof=1)) if ibi_s.size > 1 else 0.0
    diffs = np.diff(ibi_s)
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if diffs.size > 0 else 0.0
    if diffs.size > 0:
        pnn50 = float(np.mean(np.abs(diffs) > 0.05))
    else:
        pnn50 = 0.0
    cvsd = rmssd / mean_ibi if mean_ibi > 0 else 0.0
    mean_hr_from_ibi = 60.0 / mean_ibi if mean_ibi > 0 else 0.0

    return [mean_ibi, sdnn, rmssd, pnn50, cvsd, mean_hr_from_ibi]


def calculate_ibi_spectral_features(ibi_epoch):
    """
    HRV frequency-domain features computed from the IBI series (not raw BVP).
    Deduplicates held IBI values, interpolates to 4 Hz, runs Welch PSD.
    Returns 6 features: IBI_VLF, IBI_LF, IBI_HF, IBI_LF_HF, IBI_nHF, IBI_TP
    """
    ibi_s = _unique_successive(ibi_epoch)
    ibi_s = ibi_s[np.isfinite(ibi_s) & (ibi_s > 0.05) & (ibi_s < 3.0)]

    if ibi_s.size < 5:
        return [0.0] * 6

    beat_times = np.cumsum(ibi_s)
    beat_times -= beat_times[0]

    resample_fs = 4.0
    t_uniform = np.arange(0, beat_times[-1], 1.0 / resample_fs)
    if t_uniform.size < 16:
        return [0.0] * 6

    ibi_interp = np.interp(t_uniform, beat_times, ibi_s)
    ibi_interp = ibi_interp - np.mean(ibi_interp)

    nperseg = min(len(ibi_interp), 64)
    if nperseg < 8:
        return [0.0] * 6

    f, pxx = welch(ibi_interp, fs=resample_fs, nperseg=nperseg)

    vlf_mask = (f >= 0.003) & (f < 0.04)
    lf_mask = (f >= 0.04) & (f < 0.15)
    hf_mask = (f >= 0.15) & (f < 0.4)

    vlf = _band_power_trapz(pxx[vlf_mask], f[vlf_mask]) if np.any(vlf_mask) else 0.0
    lf = _band_power_trapz(pxx[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0.0
    hf = _band_power_trapz(pxx[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0.0

    total = vlf + lf + hf
    lf_hf = lf / hf if hf > 1e-12 else 0.0
    nhf = hf / total if total > 1e-12 else 0.0

    return [vlf, lf, hf, lf_hf, nhf, total]


# ─── Auxiliary sensor features ───────────────────────────────────────

def calculate_hr_features(hr_epoch):
    """4 features from HR (bpm) channel: mean, std, range, linear trend slope."""
    hr = np.asarray(hr_epoch, dtype=np.float64).ravel()
    hr = hr[np.isfinite(hr)]
    if hr.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    mean_hr = float(np.mean(hr))
    std_hr = float(np.std(hr, ddof=1)) if hr.size > 1 else 0.0
    range_hr = float(np.max(hr) - np.min(hr))
    if hr.size > 1:
        t = np.arange(hr.size, dtype=np.float64)
        slope, _, _, _, _ = stats.linregress(t, hr)
        hr_trend = float(slope)
    else:
        hr_trend = 0.0
    return [mean_hr, std_hr, range_hr, hr_trend]


def calculate_acc_features(acc_epoch):
    """7 motion features from (T, 3) accelerometer epoch (X, Y, Z)."""
    acc = np.asarray(acc_epoch, dtype=np.float64)
    if acc.ndim != 2 or acc.shape[1] != 3:
        return [0.0] * 7
    x, y, z = acc[:, 0], acc[:, 1], acc[:, 2]
    if not np.all(np.isfinite(acc)):
        acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
        x, y, z = acc[:, 0], acc[:, 1], acc[:, 2]
    mag = np.sqrt(x * x + y * y + z * z)
    mag_mean = float(np.mean(mag))
    mag_std = float(np.std(mag, ddof=1)) if mag.size > 1 else 0.0
    sx = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    sy = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    sz = float(np.std(z, ddof=1)) if z.size > 1 else 0.0
    sma = float(np.mean(np.abs(x) + np.abs(y) + np.abs(z)))
    if mag.size > 1:
        dmag = np.abs(np.diff(mag))
        thresh = max(1e-6, 0.5 * (np.median(dmag) + 1e-9))
        movement_count = int(np.sum(dmag > thresh))
    else:
        movement_count = 0
    return [mag_mean, mag_std, sx, sy, sz, float(movement_count), sma]


def calculate_eda_features(eda_epoch):
    """4 EDA features: mean, std, range, count of SCR-like peaks."""
    eda = np.asarray(eda_epoch, dtype=np.float64).ravel()
    eda = eda[np.isfinite(eda)]
    if eda.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    mean_eda = float(np.mean(eda))
    std_eda = float(np.std(eda, ddof=1)) if eda.size > 1 else 0.0
    range_eda = float(np.max(eda) - np.min(eda))
    if eda.size < 5:
        n_scr = 0
    else:
        baseline = np.median(eda)
        prominence = max(1e-6, 0.05 * (np.percentile(eda, 90) - np.percentile(eda, 10) + 1e-9))
        peaks, _ = find_peaks(eda, prominence=prominence, distance=max(1, len(eda) // 200))
        peaks = peaks[eda[peaks] > baseline]
        n_scr = int(len(peaks))
    return [mean_eda, std_eda, range_eda, float(n_scr)]


# ─── PPG statistical / nonlinear / temporal / frequency features ─────

def calculate_statistical_features(epoch):
    """16 time-domain statistical metrics for a PPG waveform array."""
    me = np.mean(epoch)
    std = np.std(epoch)
    mad = np.mean(np.abs(epoch - me))
    mabd = np.median(np.abs(epoch - np.median(epoch)))
    iqr = stats.iqr(epoch)
    rms = np.sqrt(np.mean(np.square(epoch)))

    sf = rms / me if me != 0 else 0
    tme25 = stats.trim_mean(epoch, 0.25)
    tme50 = float(np.median(epoch))
    abs_epoch = np.abs(epoch) + 1e-10
    abs_epoch = np.clip(abs_epoch, 1e-10, 1e10)
    gme = float(stats.gmean(abs_epoch))

    max_val = np.max(epoch)
    min_val = np.min(epoch)

    sk = stats.skew(epoch)
    ku = stats.kurtosis(epoch)
    ncm_3 = stats.moment(epoch, moment=3)

    acl = np.sum(np.abs(np.diff(epoch)))

    return [mad, mabd, iqr, ncm_3, acl, sf, me, std, rms, tme25, tme50, gme, max_val, min_val, sk, ku]


def _higuchi_fd(x, kmax=10):
    """Computes Higuchi Fractal Dimension (HFD)."""
    N = len(x)
    if N < 2:
        return 0.0
    L = np.zeros(kmax)
    x_val = np.zeros(kmax)
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) > 1:
                Lmk = np.sum(np.abs(np.diff(x[idx])))
                n_max = len(idx) - 1
                Lmk = Lmk * (N - 1) / (n_max * k)
                Lk.append(Lmk)
        if Lk:
            L[k - 1] = np.log(np.mean(Lk) + 1e-10)
            x_val[k - 1] = np.log(1.0 / k)
    if len(L) > 1:
        slope, _, _, _, _ = stats.linregress(x_val, L)
        return slope
    return 0.0


def _katz_fd(x):
    """Computes Katz Fractal Dimension (KFD)."""
    N = len(x)
    if N < 2:
        return 0.0
    L = np.sum(np.abs(np.diff(x)))
    d = np.max(np.abs(x - x[0]))
    if L == 0 or d == 0:
        return 0.0
    n = N - 1
    return np.log10(n) / (np.log10(d / L) + np.log10(n))


def _sample_entropy(x, m=2, r_factor=0.2, max_n=500):
    """Fast sample entropy (SampEn) on a downsampled signal."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size > max_n:
        x = x[np.linspace(0, x.size - 1, max_n, dtype=int)]
    N = x.size
    if N < m + 2:
        return 0.0
    r = r_factor * np.std(x, ddof=1)
    if r < 1e-12:
        return 0.0

    def _count_templates(length):
        templates = np.array([x[i:i + length] for i in range(N - length)])
        count = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += np.sum(dist < r)
        return count

    A = _count_templates(m + 1)
    B = _count_templates(m)
    if B == 0:
        return 0.0
    return -np.log(A / B) if A > 0 else 0.0


def calculate_nonlinear_features(epoch):
    """8 nonlinear dynamics features from PPG."""
    x = epoch[:-1]
    y = epoch[1:]
    sd1 = np.sqrt(np.var(x - y) / 2)
    sd2 = np.sqrt(np.var(x + y) / 2)
    sd1rsd2 = sd1 / sd2 if sd2 != 0 else 0

    dy = np.diff(epoch)
    ddy = np.diff(dy)
    var_zero = np.var(epoch)
    var_first = np.var(dy)
    var_second = np.var(ddy)

    hjm = np.sqrt(var_first / var_zero) if var_zero != 0 else 0
    hjc = np.sqrt(var_second / var_first) / hjm if hjm != 0 else 0

    hfd = _higuchi_fd(epoch, kmax=10)
    kfd = _katz_fd(epoch)
    sampen = _sample_entropy(epoch)

    return [sd1, sd2, sd1rsd2, hjm, hjc, hfd, kfd, sampen]


def calculate_temporal_features(epoch, fs):
    """4 timing-related features from PPG."""
    if len(epoch) == 0 or np.std(epoch) < 1e-6:
        return [0.0, 0.0, 0.0, 0.0]

    zero_crossings = np.where(np.diff(np.sign(epoch)))[0]
    zcr = len(zero_crossings) / len(epoch)

    distance_req = int(fs * 0.4) if (fs * 0.4) > 1 else 1
    peaks, _ = find_peaks(epoch, distance=distance_req)
    peak_count = len(peaks) / (len(epoch) / fs)

    ipi = np.mean(np.diff(peaks)) / fs if len(peaks) > 1 else 0.0

    troughs, _ = find_peaks(-epoch, distance=distance_req)
    rise_times = []
    if len(troughs) > 0 and len(peaks) > 0:
        for peak in peaks:
            preceding = troughs[troughs < peak]
            if len(preceding) > 0:
                rise_times.append((peak - preceding[-1]) / fs)
    mean_rise_time = np.mean(rise_times) if len(rise_times) > 0 else 0.0

    return [zcr, peak_count, ipi, mean_rise_time]


def calculate_frequency_features(epoch, fs):
    """4 BVP spectral power features using Welch's method."""
    if len(epoch) == 0 or np.std(epoch) < 1e-6:
        return [0.0, 0.0, 0.0, 0.0]

    nperseg = len(epoch)
    f, pxx = welch(epoch, fs=fs, nperseg=nperseg)

    vlf_mask = (f >= 0.003) & (f < 0.04)
    lf_mask = (f >= 0.04) & (f < 0.15)
    hf_mask = (f >= 0.15) & (f < 0.4)

    vlf_power = _band_power_trapz(pxx[vlf_mask], f[vlf_mask]) if np.any(vlf_mask) else 0.0
    lf_power = _band_power_trapz(pxx[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0.0
    hf_power = _band_power_trapz(pxx[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0.0

    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0

    return [vlf_power, lf_power, hf_power, lf_hf_ratio]


# ─── PPG morphological features (pulse wave analysis) ───────────────

def calculate_ppg_morphology_features(epoch, fs):
    """
    Pulse wave morphological features: systolic/diastolic amplitudes,
    rise/fall timing, pulse widths, and 2nd-derivative a/b ratio (APG).
    Returns 14 features.
    """
    _ZERO = [0.0] * 14
    if len(epoch) < int(fs * 1.0) or np.std(epoch) < 1e-6:
        return _ZERO

    min_dist = max(1, int(fs * 0.4))
    span = np.percentile(epoch, 95) - np.percentile(epoch, 5)
    if span < 1e-8:
        return _ZERO
    prominence = 0.1 * span

    sys_peaks, _ = find_peaks(epoch, distance=min_dist, prominence=prominence)
    troughs, _ = find_peaks(-epoch, distance=min_dist)

    if len(sys_peaks) < 2 or len(troughs) < 1:
        return _ZERO

    sys_amps, dia_amps, pulse_amps = [], [], []
    rise_ts, fall_ts, pulse_ws = [], [], []

    for pk in sys_peaks:
        pre = troughs[troughs < pk]
        post = troughs[troughs > pk]
        if len(pre) == 0 or len(post) == 0:
            continue
        onset, offset = pre[-1], post[0]
        s_amp = float(epoch[pk])
        d_amp = float(epoch[onset])
        sys_amps.append(s_amp)
        dia_amps.append(d_amp)
        pulse_amps.append(s_amp - d_amp)
        rise_ts.append((pk - onset) / fs)
        fall_ts.append((offset - pk) / fs)
        pulse_ws.append((offset - onset) / fs)

    def _ms(arr):
        if not arr:
            return 0.0, 0.0
        return float(np.mean(arr)), float(np.std(arr))

    sa_m, sa_s = _ms(sys_amps)
    da_m, da_s = _ms(dia_amps)
    pa_m, pa_s = _ms(pulse_amps)
    rt_m, rt_s = _ms(rise_ts)
    ft_m, ft_s = _ms(fall_ts)
    pw_m, pw_s = _ms(pulse_ws)

    apg = np.diff(epoch, n=2)
    ab_ratio = 0.0
    if len(apg) > int(fs * 0.5):
        apg_peaks, _ = find_peaks(apg, distance=min_dist)
        apg_troughs, _ = find_peaks(-apg, distance=min_dist)
        if len(apg_peaks) > 0 and len(apg_troughs) > 0:
            mean_a = float(np.mean(np.abs(apg[apg_peaks])))
            mean_b = float(np.mean(np.abs(apg[apg_troughs])))
            ab_ratio = mean_b / mean_a if mean_a > 1e-12 else 0.0

    return [
        sa_m, sa_s, da_m, da_s, pa_m, pa_s,
        rt_m, rt_s, ft_m, ft_s, pw_m, pw_s,
        ab_ratio, float(len(sys_amps)),
    ]


# ─── Discrete Wavelet Transform features ────────────────────────────

_DWT_LEVEL = 4
_DWT_N_FEATS = 2 * (_DWT_LEVEL + 1)


def calculate_dwt_features(epoch, wavelet="db4", level=_DWT_LEVEL):
    """
    DWT energy and std at each decomposition level.
    Returns 2*(level+1) = 10 features for level=4.
    """
    expected = 2 * (level + 1)
    try:
        import pywt
    except ImportError:
        return [0.0] * expected

    if len(epoch) < 2 ** level or np.std(epoch) < 1e-10:
        return [0.0] * expected

    try:
        coeffs = pywt.wavedec(epoch, wavelet, level=level)
    except Exception:
        return [0.0] * expected

    feats = []
    for c in coeffs:
        feats.append(float(np.sum(c ** 2)))
        feats.append(float(np.std(c)) if len(c) > 1 else 0.0)

    while len(feats) < expected:
        feats.append(0.0)
    return feats[:expected]


# ─── Composite per-epoch BVP feature vector ─────────────────────────

def _bvp_base_features_for_epoch(epoch, fs):
    """56 BVP-derived base features for one epoch."""
    stat_feats = calculate_statistical_features(epoch)            # 16
    temp_feats = calculate_temporal_features(epoch, fs)            # 4
    nl_feats = calculate_nonlinear_features(epoch)                 # 8
    freq_feats = calculate_frequency_features(epoch, fs)           # 4
    morpho_feats = calculate_ppg_morphology_features(epoch, fs)    # 14
    dwt_feats = calculate_dwt_features(epoch)                      # 10
    return stat_feats + temp_feats + nl_feats + freq_feats + morpho_feats + dwt_feats


# ─── Feature names (order must match build_feature_vector) ──────────

_BASE_NAMES = [
    # Statistical (16)
    "MAD", "MABD", "IQR", "3rd_NCM", "ACL", "Shape_Factor", "Mean",
    "Std_Dev", "RMS", "Trimmed_Mean_25", "Trimmed_Mean_50", "Geometric_Mean",
    "Max_Val", "Min_Val", "Skewness", "Kurtosis",
    # Temporal (4)
    "ZCR", "Peak_Count", "IPI", "Rise_Time",
    # Nonlinear (8)
    "Poincare_SD1", "Poincare_SD2", "SD1_to_SD2_Ratio", "Hjorth_Mobility",
    "Hjorth_Complexity", "Higuchi_FD", "Katz_FD", "SampEn",
    # BVP Frequency (4)
    "BVP_VLF_Power", "BVP_LF_Power", "BVP_HF_Power", "BVP_LF_HF_Ratio",
    # PPG Morphology (14)
    "Morph_SysAmp_Mean", "Morph_SysAmp_Std",
    "Morph_DiaAmp_Mean", "Morph_DiaAmp_Std",
    "Morph_PulseAmp_Mean", "Morph_PulseAmp_Std",
    "Morph_RiseTime_Mean", "Morph_RiseTime_Std",
    "Morph_FallTime_Mean", "Morph_FallTime_Std",
    "Morph_PulseWidth_Mean", "Morph_PulseWidth_Std",
    "Morph_APG_AB_Ratio", "Morph_ValidPulses",
    # DWT (10)
    "DWT_cA4_Energy", "DWT_cA4_Std",
    "DWT_cD4_Energy", "DWT_cD4_Std",
    "DWT_cD3_Energy", "DWT_cD3_Std",
    "DWT_cD2_Energy", "DWT_cD2_Std",
    "DWT_cD1_Energy", "DWT_cD1_Std",
    # HRV time-domain from IBI (6)
    "Mean_IBI", "SDNN_IBI", "RMSSD_IBI", "pNN50_IBI", "CVSD_IBI", "Mean_HR_from_IBI",
    # IBI spectral (6)
    "IBI_VLF", "IBI_LF", "IBI_HF", "IBI_LF_HF", "IBI_nHF", "IBI_TP",
    # HR channel (4)
    "Mean_HR", "Std_HR", "Range_HR", "HR_Trend",
    # Accelerometer (7)
    "ACC_Mag_Mean", "ACC_Mag_Std", "ACC_X_Std", "ACC_Y_Std", "ACC_Z_Std",
    "ACC_Movement_Count", "ACC_SMA",
    # EDA (4)
    "Mean_EDA", "Std_EDA", "Range_EDA", "EDA_SCR_Peaks",
]

_CTX_RMEAN_NAMES = [
    "ACC_Mag_Mean", "ACC_Movement_Count", "ACC_SMA",
    "Mean_HR", "RMSSD_IBI", "Mean_EDA", "EDA_SCR_Peaks",
]
_CTX_RSTD_NAMES = ["Mean_HR", "RMSSD_IBI"]

_BASE_NAME_TO_IDX = {n: i for i, n in enumerate(_BASE_NAMES)}
_CTX_RMEAN_IDXS = [_BASE_NAME_TO_IDX[n] for n in _CTX_RMEAN_NAMES]
_CTX_RSTD_IDXS = [_BASE_NAME_TO_IDX[n] for n in _CTX_RSTD_NAMES]

_CONTEXT_NAMES = (
    [f"Ctx_RMean_{n}" for n in _CTX_RMEAN_NAMES]
    + [f"Ctx_RStd_{n}" for n in _CTX_RSTD_NAMES]
    + ["Ctx_EpochPos"]
)

_CONTEXT_WINDOW = 5


# ─── Full feature vector builder ────────────────────────────────────

def build_feature_vector(subject_data, fs=125):
    """
    Build feature matrix for all epochs of one subject.

    Parameters
    ----------
    subject_data : np.ndarray or dict
        If ndarray: shape (num_epochs, samples_per_epoch) -- BVP only.
        If dict: ``bvp`` (n, T), optional ``hr``, ``ibi``, ``eda`` (n, T),
        ``acc`` (n, T, 3).
    fs : float
        Sampling rate (Hz) for BVP features.

    Returns
    -------
    np.ndarray of shape (n_epochs, n_features)
        base (83) + deltas (83) + temporal context (10) = 176 features.
    """
    if isinstance(subject_data, np.ndarray):
        pp = np.asarray(subject_data, dtype=np.float64)
        n_epochs, T = pp.shape
        subject_data = {
            "bvp": pp,
            "hr": np.zeros((n_epochs, T)),
            "ibi": np.zeros((n_epochs, T)),
            "eda": np.zeros((n_epochs, T)),
            "acc": np.zeros((n_epochs, T, 3)),
        }
    else:
        subject_data = dict(subject_data)
        pp = np.asarray(subject_data["bvp"], dtype=np.float64)
        n_epochs, T = pp.shape
        for key in ("hr", "ibi", "eda"):
            if key not in subject_data or subject_data[key] is None:
                subject_data[key] = np.zeros((n_epochs, T))
            else:
                subject_data[key] = np.asarray(subject_data[key], dtype=np.float64)
                if subject_data[key].shape != (n_epochs, T):
                    subject_data[key] = np.zeros((n_epochs, T))
        if "acc" not in subject_data or subject_data["acc"] is None:
            subject_data["acc"] = np.zeros((n_epochs, T, 3))
        else:
            subject_data["acc"] = np.asarray(subject_data["acc"], dtype=np.float64)
            if subject_data["acc"].shape != (n_epochs, T, 3):
                subject_data["acc"] = np.zeros((n_epochs, T, 3))

    feature_matrix = []
    for i in range(n_epochs):
        bvp_part = _bvp_base_features_for_epoch(subject_data["bvp"][i], fs)     # 56
        hrv_part = calculate_hrv_features(subject_data["ibi"][i])                # 6
        ibi_spec = calculate_ibi_spectral_features(subject_data["ibi"][i])       # 6
        hr_part = calculate_hr_features(subject_data["hr"][i])                   # 4
        acc_part = calculate_acc_features(subject_data["acc"][i])                # 7
        eda_part = calculate_eda_features(subject_data["eda"][i])                # 4
        feature_matrix.append(
            bvp_part + hrv_part + ibi_spec + hr_part + acc_part + eda_part
        )

    base_features = np.array(feature_matrix, dtype=np.float64)

    deltas = np.zeros_like(base_features)
    if base_features.shape[0] > 1:
        deltas[1:] = base_features[1:] - base_features[:-1]
        deltas[0] = deltas[1]

    w = _CONTEXT_WINDOW
    n_ctx = len(_CTX_RMEAN_IDXS) + len(_CTX_RSTD_IDXS) + 1
    context = np.zeros((n_epochs, n_ctx), dtype=np.float64)

    for i in range(n_epochs):
        lo = max(0, i - w)
        hi = min(n_epochs, i + w + 1)
        window = base_features[lo:hi]
        col = 0
        for idx in _CTX_RMEAN_IDXS:
            context[i, col] = float(np.mean(window[:, idx]))
            col += 1
        for idx in _CTX_RSTD_IDXS:
            context[i, col] = float(np.std(window[:, idx]))
            col += 1
        context[i, col] = i / max(1, n_epochs - 1)

    final_features = np.hstack((base_features, deltas, context))
    final_features = np.nan_to_num(final_features, nan=0.0, posinf=0.0, neginf=0.0)
    return final_features


def get_feature_names():
    """Ordered list of all feature names (base + delta + context)."""
    delta_names = [f"Delta_{n}" for n in _BASE_NAMES]
    return list(_BASE_NAMES) + delta_names + list(_CONTEXT_NAMES)
