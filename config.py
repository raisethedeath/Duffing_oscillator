# config.py
import numpy as np

class Config:
    # 默认物理参数
    DEFAULT_PARAMS = {
        'delta': 0.2,   # 阻尼
        'alpha': 1.0,   # 线性刚度
        'beta': 1.0,    # 非线性刚度
        'gamma': 2.5,   # 驱动力幅值
        'omega': 2.0    # 驱动频率
    }

    # 模拟设置
    SIM_SETTINGS = {
        'dt': 0.01,         # 时间步长
        't_max_traj': 150,  # 单条轨迹时长
        't_max_scan': 150,  # 扫描时的时长 (为了速度稍微短一点)
        't_max_chaos': 3000 # 混沌模拟时长
    }

    # 扫描范围设置 (用于 Task 2)
    # 为了演示速度，网格设为 10x10，实际研究可设为 50x50
    SCAN_RANGES = {
        'delta': np.linspace(0.0, 0.5, 50),
        'alpha': np.linspace(-0.5, 1.5, 50),
        'beta':  np.linspace(0.0, 1.5, 50),
        'gamma': np.linspace(1.5, 3.0, 50),
        'omega': np.linspace(0.5, 5.0, 50)
    }

    # 频率扫描设置 (用于 Task 3)
    FREQ_SWEEP = {
        'w_start': 0.5,
        'w_end': 5.5,
        'steps': 60
    }
