

"""
信号预处理模块
该模块提供了多种生理信号（主要是EEG和EMG）的预处理功能

主要功能包括：
- 滤波处理（带通、陷波）
- 基线矫正
- 数据加载（EDF、CSV格式）
- 异常通道检测与修复
- ICA伪迹剔除
- 完整预处理流水线
"""
from typing import Tuple, List, Optional
import numpy as np
from scipy import signal

# 尝试导入mne库（用于高级信号处理，如ICA），若未安装则设置为None
try:
    import mne
except Exception:
    mne = None


def bandpass_filter(x: np.ndarray, fs: float, low: float = 1.0, high: float = 45.0, order: int = 5) -> np.ndarray:
    """
    带通滤波 - 保留指定频率范围内的信号成分

    参数:
    x: np.ndarray - 输入信号数组 (形状: (n_channels, n_samples) 或 (n_samples,))
    fs: float - 采样频率 (Hz)
    low: float - 低频截止频率，默认1.0 Hz
    high: float - 高频截止频率，默认45.0 Hz
    order: int - 滤波器阶数，默认5

    返回:
    np.ndarray - 滤波后的信号（与输入形状相同）
    """
    # 计算归一化截止频率
    nyquist = 0.5 * fs
    low_norm = low / nyquist
    high_norm = high / nyquist
    
    # 设计Butterworth带通滤波器
    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    
    # 使用零相位滤波（filtfilt）避免相位失真
    return signal.filtfilt(b, a, x, axis=-1)


def notch_filter(x: np.ndarray, fs: float, freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
    """
    陷波滤波 - 去除特定频率的干扰（通常是50Hz或60Hz工频干扰）

    参数:
    x: np.ndarray - 输入信号数组 (形状: (n_channels, n_samples) 或 (n_samples,))
    fs: float - 采样频率 (Hz)
    freq: float - 要去除的干扰频率，默认50.0 Hz
    quality: float - 滤波器品质因子，值越大带宽越窄，默认30.0

    返回:
    np.ndarray - 滤波后的信号（与输入形状相同）
    """
    # 计算归一化中心频率
    w0 = freq / (0.5 * fs)
    
    # 设计IIR陷波滤波器
    b, a = signal.iirnotch(w0, quality)
    
    # 使用零相位滤波
    return signal.filtfilt(b, a, x, axis=-1)


def baseline_correction(x: np.ndarray) -> np.ndarray:
    """
    基线矫正 - 去除信号中的线性趋势

    参数:
    x: np.ndarray - 输入信号数组 (形状: (n_channels, n_samples) 或 (n_samples,))

    返回:
    np.ndarray - 矫正后的信号（与输入形状相同）
    """
    # 使用线性 detrending 去除基线漂移
    return signal.detrend(x, axis=-1)


def load_edf(path: str):
    """
    加载EDF格式的脑电数据文件

    参数:
    path: str - EDF文件路径

    返回:
    tuple - (数据数组, 采样频率, 通道名称列表)
        data: np.ndarray - 数据数组，形状 (n_channels, n_samples)
        sfreq: float - 采样频率 (Hz)
        ch_names: list - 通道名称列表
    """
    # 检查mne库是否已安装
    if mne is None:
        raise ImportError('mne is required to read EDF files. Install mne.')
    
    # 使用mne加载EDF文件
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    
    # 获取数据、采样频率和通道名称
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    
    return data, sfreq, ch_names


def load_csv(path: str, delimiter: str = ',') -> Tuple[np.ndarray, float, List[str]]:
    """
    加载CSV格式的数据文件

    参数:
    path: str - CSV文件路径
    delimiter: str - 分隔符，默认','

    返回:
    tuple - (数据数组, 采样频率, 通道名称列表)
        data: np.ndarray - 数据数组，形状 (n_channels, n_samples)
        sfreq: float - 采样频率 (Hz)，当前版本返回None，需要用户手动指定
        ch_names: list - 通道名称列表
    """
    import pandas as pd
    
    # 读取CSV文件
    df = pd.read_csv(path, delimiter=delimiter)
    
    # 检查文件列数是否足够
    if df.shape[1] < 2:
        raise ValueError('CSV 文件列数过少')
    
    # 转换为numpy数组并转置（确保形状为(n_channels, n_samples)）
    arr = df.values.T
    ch_names = list(df.columns)
    
    # 当前版本无法自动推断采样率，返回None
    return arr, None, ch_names


def detect_bad_channels(data: np.ndarray, fs: float, flat_threshold: float = 1e-6, z_thresh: float = 3.0) -> List[int]:
    """
    检测异常通道（坏导）

    参数:
    data: np.ndarray - 数据数组，形状 (n_channels, n_samples)
    fs: float - 采样频率 (Hz)
    flat_threshold: float - 平坦通道的标准差阈值，默认1e-6
    z_thresh: float - 异常通道检测的Z分数阈值，默认3.0

    返回:
    list - 异常通道的索引列表
    """
    # 计算各通道的标准差
    stds = np.std(data, axis=1)
    bad = []
    
    # 检测平坦通道（标准差过小或为NaN）
    for i, s in enumerate(stds):
        if s < flat_threshold or np.isnan(s):
            bad.append(i)
    
    # 使用Z分数检测极端值通道
    z = (stds - np.mean(stds)) / np.std(stds)
    bad.extend(list(np.where(np.abs(z) > z_thresh)[0]))
    
    # 返回去重并排序后的异常通道索引
    return sorted(list(set(bad)))


def interpolate_bad_channels(data: np.ndarray, bad_idx: List[int]) -> np.ndarray:
    """
    插值修复异常通道

    参数:
    data: np.ndarray - 数据数组，形状 (n_channels, n_samples)
    bad_idx: list - 异常通道的索引列表

    返回:
    np.ndarray - 修复后的数据数组，形状 (n_channels, n_samples)
    """
    out = data.copy()
    n_channels = data.shape[0]
    
    # 确定正常通道索引
    good_idx = [i for i in range(n_channels) if i not in bad_idx]
    
    # 检查是否有可用的正常通道
    if len(good_idx) == 0:
        raise ValueError('没有可用通道用于插值')
    
    # 计算正常通道的平均信号
    mean_signal = np.mean(data[good_idx, :], axis=0)
    
    # 使用平均信号替换异常通道
    for i in bad_idx:
        out[i, :] = mean_signal
    
    return out


def run_ica_remove(raw_data: np.ndarray, sfreq: float, picks: Optional[List[int]] = None, n_components: int = 20) -> np.ndarray:
    """
    使用MNE ICA去除伪迹（若安装了mne库）

    参数:
    raw_data: np.ndarray - 原始数据数组，形状 (n_channels, n_samples)
    sfreq: float - 采样频率 (Hz)
    picks: Optional[List[int]] - 要处理的通道索引列表，默认None（处理所有通道）
    n_components: int - ICA组件数量，默认20

    返回:
    np.ndarray - 去除伪迹后的数据数组，形状 (n_channels, n_samples)
    """
    # 检查mne库是否已安装
    if mne is None:
        raise ImportError('mne 必需以运行 ICA。请安装 mne。')
    
    n_channels, n_samples = raw_data.shape
    
    # 创建MNE信息对象
    info = mne.create_info(
        ch_names=[f'EEG{i}' for i in range(n_channels)],
        sfreq=sfreq,
        ch_types=['eeg'] * n_channels
    )
    
    # 创建MNE Raw对象
    raw = mne.io.RawArray(raw_data, info)
    
    # 初始化ICA对象
    ica = mne.preprocessing.ICA(
        n_components=min(n_components, n_channels - 1),
        random_state=42
    )
    
    # 拟合ICA模型
    ica.fit(raw)
    
    # 注意：这里需要用户手动指定要排除的伪迹组件
    # 示例：ica.exclude = [0, 1]  # 排除前两个组件
    # 这里仅提供空实现，用户需要根据实际情况调整
    
    # 应用ICA去除伪迹
    ica.apply(raw)
    
    # 返回处理后的数据
    return raw.get_data()


def preprocess_pipeline(data: np.ndarray, sfreq: float, do_notch: bool = True, bad_channel_handling: bool = True) -> np.ndarray:
    """
    预处理流水线 - 整合多种预处理步骤

    参数:
    data: np.ndarray - 输入数据数组，形状 (n_channels, n_samples)
    sfreq: float - 采样频率 (Hz)
    do_notch: bool - 是否执行陷波滤波，默认True
    bad_channel_handling: bool - 是否检测和修复异常通道，默认True

    返回:
    np.ndarray - 预处理后的数据数组，形状 (n_channels, n_samples)
    """
    # 创建数据副本
    x = data.copy()
    
    # 逐通道进行带通滤波和陷波滤波
    for i in range(x.shape[0]):
        # 带通滤波（1-45Hz）
        x[i, :] = bandpass_filter(x[i, :], fs=sfreq, low=1.0, high=45.0)
        
        # 陷波滤波（去除50Hz干扰）
        if do_notch:
            x[i, :] = notch_filter(x[i, :], fs=sfreq, freq=50.0)
    
    # 基线矫正
    x = baseline_correction(x)
    
    # 异常通道检测与修复
    if bad_channel_handling:
        bad = detect_bad_channels(x, fs=sfreq)
        if len(bad) > 0:
            x = interpolate_bad_channels(x, bad)
    
    return x

"""
疲劳指标计算模块
该模块提供了多种基于脑电(EEG)和肌电(EMG)信号的疲劳指标计算功能

主要功能包括：
- 频段功率计算
- 相对功率比
- θ/α 比值（常用脑电疲劳指标）
- EMG K-index（肌电疲劳指标）
- 综合疲劳指数计算
"""
from typing import Tuple
import numpy as np
from scipy import signal


def bandpower(x: np.ndarray, fs: float, band: Tuple[float, float], nperseg: int = 512) -> float:
    """
    计算信号在指定频段的功率

    参数:
    x: np.ndarray - 输入信号数组
    fs: float - 采样频率 (Hz)
    band: Tuple[float, float] - 频段范围 (low, high) Hz
    nperseg: int - Welch方法的分段长度，默认512

    返回:
    float - 指定频段的功率值
    """
    # 使用Welch方法计算功率谱密度(PSD)
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg)
    # 选择指定频段的频率索引
    idx = np.logical_and(f >= band[0], f <= band[1])
    # 使用梯形法计算频段内的功率积分
    return np.trapz(Pxx[idx], f[idx])


def relative_band_powers(x: np.ndarray, fs: float) -> dict:
    """
    计算各频段的绝对功率和相对功率

    参数:
    x: np.ndarray - 输入信号数组
    fs: float - 采样频率 (Hz)

    返回:
    dict - 包含绝对功率和相对功率的字典
        'absolute': 各频段的绝对功率 {'delta': float, 'theta': float, ...}
        'relative': theta、alpha、beta频段的相对功率（相对于这三个频段总和）
    """
    # 定义各频段范围
    bands = {
        'delta': (0.5, 3),    # δ波：0.5-3Hz
        'theta': (4, 7),      # θ波：4-7Hz
        'alpha': (8, 13),     # α波：8-13Hz
        'beta': (14, 30),     # β波：14-30Hz
        'gamma': (31, 45)     # γ波：31-45Hz
    }
    
    # 计算各频段的绝对功率
    powers = {k: bandpower(x, fs, b) for k, b in bands.items()}
    
    # 计算theta、alpha、beta频段的总功率
    total = powers['theta'] + powers['alpha'] + powers['beta']
    
    # 计算相对功率（防止除以零）
    rel = {k: (powers[k] / total if total > 0 else 0.0) for k in ['theta', 'alpha', 'beta']}
    
    return {'absolute': powers, 'relative': rel}


def theta_alpha_ratio(x: np.ndarray, fs: float) -> float:
    """
    计算θ/α比值（常用的脑电疲劳指标）

    参数:
    x: np.ndarray - 输入脑电信号
    fs: float - 采样频率 (Hz)

    返回:
    float - θ/α比值，值越大表示疲劳程度越高
    """
    # 计算theta频段功率
    p_theta = bandpower(x, fs, (4, 7))
    # 计算alpha频段功率
    p_alpha = bandpower(x, fs, (8, 13))
    # 返回比值（防止除以零，若alpha功率为0则返回无穷大）
    return (p_theta / p_alpha) if p_alpha > 0 else np.inf


def emg_k_index(x: np.ndarray, fs: float) -> float:
    """
    计算EMG K-index（肌电疲劳指标）

    参数:
    x: np.ndarray - 输入肌电信号
    fs: float - 采样频率 (Hz)

    返回:
    float - K-index值，值越小表示疲劳程度越高
    """
    # 使用Welch方法计算功率谱密度
    f, Pxx = signal.welch(x, fs=fs, nperseg=1024)
    
    # 检查功率谱是否全为0
    if np.sum(Pxx) == 0:
        return 0.0
    
    # 计算平均频率
    mean_freq = np.sum(f * Pxx) / np.sum(Pxx)
    # 计算峰值频率
    peak_freq = f[np.argmax(Pxx)]
    
    # 返回峰值频率与平均频率的比值（K-index）
    return float(peak_freq / mean_freq) if mean_freq > 0 else 0.0


def normalize_to_0_1(val: float, vmin: float, vmax: float) -> float:
    """
    将数值归一化到0-1范围

    参数:
    val: float - 输入值
    vmin: float - 最小值（归一化后对应0）
    vmax: float - 最大值（归一化后对应1）

    返回:
    float - 归一化后的值（0-1之间）
    """
    # 处理无穷大值
    if np.isinf(val):
        return 1.0
    # 归一化并限制在0-1范围内
    return float(np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0))


def compute_fatigue_index(eeg_channel: np.ndarray, emg_channel: np.ndarray, fs_eeg: float, fs_emg: float) -> float:
    """
    计算综合疲劳指数

    参数:
    eeg_channel: np.ndarray - 脑电信号通道
    emg_channel: np.ndarray - 肌电信号通道
    fs_eeg: float - 脑电信号采样频率 (Hz)
    fs_emg: float - 肌电信号采样频率 (Hz)

    返回:
    float - 综合疲劳指数，值越大表示疲劳程度越高
    """
    # 计算θ/α比值
    tar = theta_alpha_ratio(eeg_channel, fs_eeg)
    
    # 将θ/α比值归一化到0-1范围（经验值范围：0.5-5.0）
    norm_tar = normalize_to_0_1(tar, 0.5, 5.0)
    
    # 计算EMG K-index
    k = emg_k_index(emg_channel, fs_emg)
    
    # 综合计算疲劳指数（公式：k + 0.9 * 归一化θ/α）
    # 注：K-index通常范围接近(0,0.1)，若超出此范围可能影响结果准确性
    return k + 0.9 * norm_tar

"""
示例：生成合成EEG/EMG信号，执行预处理并计算疲劳指标

这个示例演示了如何使用项目的预处理和疲劳指标计算功能，
包括信号生成、预处理流水线、指标计算和可视化。
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从src模块导入所需功能
from src import preprocessing, metrics


def generate_synthetic_eeg(n_channels=8, duration=10.0, fs=256):
    """生成合成EEG信号"""
    t = np.arange(0, duration, 1/fs)
    sig = np.zeros((n_channels, len(t)))
    for ch in range(n_channels):
        # 基本 alpha(10Hz) + theta(6Hz) + 噪声
        sig[ch] = 20e-6 * np.sin(2*np.pi*10*t) + 10e-6 * np.sin(2*np.pi*6*t) + 5e-6 * np.random.randn(len(t))
    return sig, fs


def generate_synthetic_emg(duration=10.0, fs=1000):
    """生成合成EMG信号"""
    t = np.arange(0, duration, 1/fs)
    # 高频成分
    sig = 0.01 * np.random.randn(len(t)) + 0.005 * np.sin(2*np.pi*80*t)
    return sig, fs


if __name__ == '__main__':
    # 1. 生成合成信号
    print("正在生成合成EEG和EMG信号...")
    eeg, fs_eeg = generate_synthetic_eeg(n_channels=8, duration=30.0, fs=256)
    emg, fs_emg = generate_synthetic_emg(duration=30.0, fs=1000)

    # 2. 执行预处理
    print("正在执行信号预处理...")
    pre = preprocessing.preprocess_pipeline(eeg, sfreq=fs_eeg)
    
    # 3. 选择感兴趣的通道进行分析
    eeg_ch0 = pre[0]  # 第0个EEG通道
    emg_sig = emg     # EMG信号

    # 4. 计算疲劳指标
    print("正在计算疲劳指标...")
    tar = metrics.theta_alpha_ratio(eeg_ch0, fs_eeg)
    k = metrics.emg_k_index(emg_sig, fs_emg)
    fatigue = metrics.compute_fatigue_index(eeg_ch0, emg_sig, fs_eeg, fs_emg)

    # 5. 输出结果
    print("\n--- 疲劳指标计算结果 ---")
    print(f'Theta/Alpha 比值: {tar:.3f}')
    print(f'EMG k-index: {k:.4f}')
    print(f'综合疲劳指数: {fatigue:.3f}')

    # 6. 可视化信号
    print("\n正在显示信号波形...")
    plt.figure(figsize=(12, 8))
    
    # 显示预处理后的EEG信号
    plt.subplot(2, 1, 1)
    plt.plot(eeg_ch0[:500])  # 只显示前500个样本
    plt.title('Preprocessed EEG (Channel 0)')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # 显示EMG信号
    plt.subplot(2, 1, 2)
    plt.plot(emg_sig[:2000])  # 只显示前2000个样本
    plt.title('Synthetic EMG Signal')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\n示例运行完成！")