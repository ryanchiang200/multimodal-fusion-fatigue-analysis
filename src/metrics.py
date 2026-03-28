"""疲劳指标计算：相对功率、θ/α、EMG k-index 等"""
from typing import Tuple
import numpy as np
from scipy import signal


def bandpower(x: np.ndarray, fs: float, band: Tuple[float, float], nperseg: int = 512) -> float:
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg)
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx], f[idx])


def relative_band_powers(x: np.ndarray, fs: float) -> dict:
    bands = {'delta': (0.5, 3), 'theta': (4, 7), 'alpha': (8, 13), 'beta': (14, 30), 'gamma': (31, 45)}
    powers = {k: bandpower(x, fs, b) for k, b in bands.items()}
    total = powers['theta'] + powers['alpha'] + powers['beta']
    rel = {k: (powers[k] / total if total > 0 else 0.0) for k in ['theta', 'alpha', 'beta']}
    return {'absolute': powers, 'relative': rel}


def theta_alpha_ratio(x: np.ndarray, fs: float) -> float:
    p_theta = bandpower(x, fs, (4, 7))
    p_alpha = bandpower(x, fs, (8, 13))
    return (p_theta / p_alpha) if p_alpha > 0 else np.inf


def emg_k_index(x: np.ndarray, fs: float) -> float:
    # k = peak_freq / mean_freq
    f, Pxx = signal.welch(x, fs=fs, nperseg=1024)
    if np.sum(Pxx) == 0:
        return 0.0
    mean_freq = np.sum(f * Pxx) / np.sum(Pxx)
    peak_freq = f[np.argmax(Pxx)]
    return float(peak_freq / mean_freq) if mean_freq > 0 else 0.0


def normalize_to_0_1(val: float, vmin: float, vmax: float) -> float:
    if np.isinf(val):
        return 1.0
    return float(np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0))


def compute_fatigue_index(eeg_channel: np.ndarray, emg_channel: np.ndarray, fs_eeg: float, fs_emg: float) -> float:
    # Fatigue = k + 0.9 * Norm(theta/alpha)
    tar = theta_alpha_ratio(eeg_channel, fs_eeg)
    norm_tar = normalize_to_0_1(tar, 0.5, 5.0)  # vmin/vmax 可调
    k = emg_k_index(emg_channel, fs_emg)
    # 假定 k 范围接近 (0,0.1)，若超出会影响结果
    return k + 0.9 * norm_tar
