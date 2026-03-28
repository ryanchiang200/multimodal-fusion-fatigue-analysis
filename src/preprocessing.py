"""预处理模块：滤波、陷波、基线矫正、坏导检测、ICA伪迹剔除

依赖：numpy, scipy, mne (可选), neurokit2
"""
from typing import Tuple, List, Optional
import numpy as np
from scipy import signal

try:
    import mne
except Exception:
    mne = None


def bandpass_filter(x: np.ndarray, fs: float, low: float = 1.0, high: float = 45.0, order: int = 5) -> np.ndarray:
    b, a = signal.butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return signal.filtfilt(b, a, x, axis=-1)


def notch_filter(x: np.ndarray, fs: float, freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
    w0 = freq / (0.5 * fs)
    b, a = signal.iirnotch(w0, quality)
    return signal.filtfilt(b, a, x, axis=-1)


def baseline_correction(x: np.ndarray) -> np.ndarray:
    return signal.detrend(x, axis=-1)


def load_edf(path: str):
    if mne is None:
        raise ImportError('mne is required to read EDF files. Install mne.')
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    return data, sfreq, ch_names


def load_csv(path: str, delimiter: str = ',') -> Tuple[np.ndarray, float, List[str]]:
    import pandas as pd
    df = pd.read_csv(path, delimiter=delimiter)
    # 假设第一列为时间或样本序号，可根据实际文件调整
    if df.shape[1] < 2:
        raise ValueError('CSV 文件列数过少')
    # 如果有时间列，尝试推断采样率
    arr = df.values.T
    ch_names = list(df.columns)
    # 用户需提供采样率，示例返回 None
    return arr, None, ch_names


def detect_bad_channels(data: np.ndarray, fs: float, flat_threshold: float = 1e-6, z_thresh: float = 3.0) -> List[int]:
    # data shape: (n_channels, n_samples)
    stds = np.std(data, axis=1)
    bad = []
    for i, s in enumerate(stds):
        if s < flat_threshold or np.isnan(s):
            bad.append(i)
    # 极端值检测
    z = (stds - np.mean(stds)) / np.std(stds)
    bad.extend(list(np.where(np.abs(z) > z_thresh)[0]))
    return sorted(list(set(bad)))


def interpolate_bad_channels(data: np.ndarray, bad_idx: List[int]) -> np.ndarray:
    out = data.copy()
    n_channels = data.shape[0]
    good_idx = [i for i in range(n_channels) if i not in bad_idx]
    if len(good_idx) == 0:
        raise ValueError('没有可用通道用于插值')
    mean_signal = np.mean(data[good_idx, :], axis=0)
    for i in bad_idx:
        out[i, :] = mean_signal
    return out


def run_ica_remove(raw_data: np.ndarray, sfreq: float, picks: Optional[List[int]] = None, n_components: int = 20) -> np.ndarray:
    """使用 MNE ICA 去除伪迹（若安装了 mne）
    raw_data: (n_channels, n_samples)
    返回修正后的数据同形状
    """
    if mne is None:
        raise ImportError('mne 必需以运行 ICA。请安装 mne。')
    n_channels, n_samples = raw_data.shape
    info = mne.create_info(ch_names=[f'EEG{i}' for i in range(n_channels)], sfreq=sfreq, ch_types=['eeg'] * n_channels)
    raw = mne.io.RawArray(raw_data, info)
    ica = mne.preprocessing.ICA(n_components=min(n_components, n_channels - 1), random_state=42)
    ica.fit(raw)
    # 自动查找 EOG/ECG 伪迹（需传入参考通道或手动指定），这里仅提供空实现供用户手动选择
    # 用户可调用 ica.exclude = [idx,...] 并执行 ica.apply(raw)
    ica.apply(raw)
    return raw.get_data()


def preprocess_pipeline(data: np.ndarray, sfreq: float, do_notch: bool = True, bad_channel_handling: bool = True) -> np.ndarray:
    """简单流水线：带通 -> 陷波 -> 基线去趋势 -> 坏导处理(插值)"""
    x = data.copy()
    # bandpass per channel
    for i in range(x.shape[0]):
        x[i, :] = bandpass_filter(x[i, :], fs=sfreq, low=1.0, high=45.0)
        if do_notch:
            x[i, :] = notch_filter(x[i, :], fs=sfreq, freq=50.0)
    x = baseline_correction(x)
    if bad_channel_handling:
        bad = detect_bad_channels(x, fs=sfreq)
        if len(bad) > 0:
            x = interpolate_bad_channels(x, bad)
    return x
