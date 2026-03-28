项目：游戏疲劳监测 — 多模态生理信号预处理与疲劳指标
快速开始：
1. 创建并激活Python虚拟环境（建议Python 3.8+）

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. 运行示例（生成合成信号并执行预处理与指标计算）：

```bash
python examples/example_preprocess.py
```

说明：主要模块位于 `src/`，包括信号预处理（滤波、陷波、基线矫正、坏导检测、ICA伪迹剔除）和疲劳指标计算（θ/α、相对波段功率、EMG k-index）。

关键参考文献（供方法与指标依据）
1. Makeig, S., Bell, A. J., Jung, T.-P., & Sejnowski, T. J. (1996). Independent component analysis of electroencephalographic data. Advances in Neural Information Processing Systems, 8, 145–151.
2. Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance: a review and analysis. Brain Research Reviews, 29(2-3), 169–195.
3. Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. IEEE Transactions on Rehabilitation Engineering, 8(4), 441–446. (CSP 方法)
4. Jung, T.-P., Makeig, S., Humphries, C., Lee, T.-W., Mckeown, M. J., Iragui, V., & Sejnowski, T. J. (2000). Removing electroencephalographic artifacts by blind source separation. Psychophysiology, 37(2), 163–178.
5. De Luca, C. J. (1984). Myoelectrical manifestations of localized muscular fatigue in humans. Critical Reviews in Biomedical Engineering, 11(4), 251–279.
6. Widmann, A., Schröger, E., & Maess, B. (2015). Digital filter design for electrophysiological data — a practical approach. Journal of Neuroscience Methods, 250, 34–46.

说明：以上文献覆盖ICA伪迹去除、CSP空间滤波、EEG频带与认知/疲劳关系、EMG疲劳特征与滤波设计等方法论基础。
