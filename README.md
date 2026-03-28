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
