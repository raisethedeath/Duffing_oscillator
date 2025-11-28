# analyzer.py
import numpy as np
import pandas as pd

class Analyzer:
    @staticmethod
    def calculate_period_error(df, omega):
        """
        计算周期误差 E(t) = |x(t) - x(t-T)|
        返回: time_points, error_values, moving_avg_error
        """
        t = df['time'].values
        x = df['x'].values
        dt = t[1] - t[0]
        period = 2 * np.pi / omega
        points_per_cycle = int(np.round(period / dt))
        
        if points_per_cycle == 0 or points_per_cycle >= len(x):
            return t, np.zeros_like(t), np.zeros_like(t)

        errors = []
        valid_times = []
        
        # 从一个周期后开始计算
        for i in range(points_per_cycle, len(x)):
            err = np.abs(x[i] - x[i - points_per_cycle])
            errors.append(err)
            valid_times.append(t[i])
            
        errors = np.array(errors)
        valid_times = np.array(valid_times)
        
        # 计算移动平均 (窗口为一个周期)
        window = points_per_cycle
        moving_avg = np.convolve(errors, np.ones(window)/window, mode='valid')
        # 对齐时间轴
        ma_times = valid_times[window-1:]
        
        return valid_times, errors, ma_times, moving_avg

    @staticmethod
    def get_steady_state_time(df, omega, threshold=1e-1, window_cycles=5):
        """确定进入稳态的时间"""
        _, _, ma_times, ma_errors = Analyzer.calculate_period_error(df, omega)
        
        # 找到第一个连续低于阈值的时间点
        for i, err in enumerate(ma_errors):
            if err < threshold:
                # 简单检查后续是否也稳定
                if np.all(ma_errors[i:i+100] < threshold): 
                    return ma_times[i]
        return df['time'].iloc[-1] # 未达到稳态

    @staticmethod
    def get_poincare_points(df, omega):
        """提取庞加莱截面点"""
        t = df['time'].values
        period = 2 * np.pi / omega
        # 简单的最近邻插值提取
        indices = []
        # 忽略前 50% 的数据以去除瞬态
        start_idx = int(len(t) * 0.5)
        
        for n in range(1, int(t[-1]/period)):
            target_t = n * period
            # 在 start_idx 之后寻找
            if target_t < t[start_idx]: continue
            
            idx = (np.abs(t - target_t)).argmin()
            indices.append(idx)
            
        return df.iloc[indices]
