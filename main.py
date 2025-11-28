# main.py
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm  
# from math import sqrt

from config import Config
from solver import DuffingSolver
from analyzer import Analyzer

# 确保数据目录存在
if not os.path.exists('Data'): os.makedirs('Data')

class ComputationManager:
    def __init__(self):
        self.base_params = Config.DEFAULT_PARAMS.copy()
        self.settings = Config.SIM_SETTINGS

    def run_task1_trajectory(self):
        print(">>> Computing Task 1: Trajectory & Error Analysis...")
        solver = DuffingSolver(self.base_params)
        df = solver.solve(self.settings['t_max_traj'], self.settings['dt'])
        
        # 计算误差分析数据
        t_err, err, t_ma, ma = Analyzer.calculate_period_error(df, self.base_params['omega'])
        
        # 保存
        df.to_csv('Data/task1_traj.csv', index=False)
        np.savez('Data/task1_error.npz', t_err=t_err, err=err, t_ma=t_ma, ma=ma)

    def run_task2_parameter_scan(self):
        print(">>> Computing Task 2: Pairwise Parameter Scan (Heatmaps)...")
        # 获取所有参数名
        param_names = ['delta', 'alpha', 'beta', 'gamma', 'omega']
        # 生成两两组合: C(5,2) = 10 组
        pairs = list(itertools.combinations(param_names, 2))
        
        for p1_name, p2_name in pairs:
            print(f"   Scanning pair: {p1_name} vs {p2_name}")
            
            range1 = Config.SCAN_RANGES[p1_name]
            range2 = Config.SCAN_RANGES[p2_name]
            
            results = []
            
            # 遍历网格
            for v1 in tqdm(range1, leave=False):
                for v2 in range2:
                    # 构造当前参数集
                    current_params = self.base_params.copy()
                    current_params[p1_name] = v1
                    current_params[p2_name] = v2
                    
                    solver = DuffingSolver(current_params)
                    df = solver.solve(self.settings['t_max_scan'], self.settings['dt'])
                    if v2 != 'omega':
                        steady_time = Analyzer.get_steady_state_time(df, current_params['omega'])
                    else:
                        steady_time = Analyzer.get_steady_state_time (df, v2)
                    results.append({p1_name: v1, p2_name: v2, 'steady_time': steady_time})
            
            # 保存该组合的结果
            pd.DataFrame(results).to_csv(f'Data/task2_scan_{p1_name}_{p2_name}.csv', index=False)

    def run_task3_hysteresis(self):
        print(">>> Computing Task 3: Hysteresis Sweep...")
        w_start = Config.FREQ_SWEEP['w_start']
        w_end = Config.FREQ_SWEEP['w_end']
        steps = Config.FREQ_SWEEP['steps']
        
        w_fwd = np.linspace(w_start, w_end, steps)
        w_bwd = np.linspace(w_end, w_start, steps)
        total_time_elapsed = 0.0
        trajectory_scan_frequency = []
        
        # 正向
        res_fwd = []
        curr_x, curr_v = 0.0, 0.0
        for w in tqdm(w_fwd, desc="Forward"):
            p = self.base_params.copy()
            p['omega'] = w
            solver = DuffingSolver(p)
            df = solver.solve(100, 0.05, x0=curr_x, v0=curr_v) # 较短时间即可
            df['time'] +=total_time_elapsed
            trajectory_scan_frequency.append(df)
            total_time_elapsed += 100
            # trajectory_scan_frequency=pd.concat(trajectory_scan_frequency, df, ignore_index=True)
            
            # 取后段振幅
            steady = df.iloc[-int(len(df)*0.3):]
            amp = (steady['x'].max() - steady['x'].min()) / 2
            res_fwd.append({'omega': w, 'amp': amp})
            curr_x, curr_v = df.iloc[-1]['x'], df.iloc[-1]['v']
            
        # 反向
        res_bwd = []
        # curr_x, curr_v 继承
        for w in tqdm(w_bwd, desc="Backward"):
            p = self.base_params.copy()
            p['omega'] = w
            solver = DuffingSolver(p)
            df = solver.solve(100, 0.05, x0=curr_x, v0=curr_v)
            df['time'] += total_time_elapsed
            trajectory_scan_frequency.append(df)
            total_time_elapsed += 100
            # trajectory_scan_frequency=pd.concat(trajectory_scan_frequency, df, ignore_index=True)
            
            steady = df.iloc[-int(len(df)*0.3):]
            amp = (steady['x'].max() - steady['x'].min()) / 2
            res_bwd.append({'omega': w, 'amp': amp})
            curr_x, curr_v = df.iloc[-1]['x'], df.iloc[-1]['v']
            
        pd.DataFrame(res_fwd).to_csv('Data/task3_fwd.csv', index=False)
        pd.DataFrame(res_bwd).to_csv('Data/task3_bwd.csv', index=False)
        # print(trajectory_scan_frequency)
        trajectory_scan_frequency = pd.concat(trajectory_scan_frequency, ignore_index=True)
        pd.DataFrame(trajectory_scan_frequency).to_csv('Data/task3_trajectory_scan_frequency.csv', index=False, encoding='utf-8')

    def run_task4_chaos(self):
        print(">>> Computing Task 4: Chaos & Poincare...")
        # 混沌参数
        chaos_params = self.base_params.copy()
        chaos_params['delta'] = 0.1
        chaos_params['gamma'] = 12.0
        chaos_params['omega'] = 1.0
        chaos_params['beta'] = 0.25 # 经典Duffing混沌参数
        
        solver = DuffingSolver(chaos_params)
        df = solver.solve(self.settings['t_max_chaos'], 0.01) # 需要高精度
        
        # 提取庞加莱点
        poincare_df = Analyzer.get_poincare_points(df, chaos_params['omega'])
        
        df.to_csv('Data/task4_traj.csv', index=False)
        poincare_df.to_csv('Data/task4_poincare.csv', index=False)


class VisualizationManager:
    def plot_task1(self):
        print("Plotting Task 1...")
        df = pd.read_csv('Data/task1_traj.csv')
        err_data = np.load('Data/task1_error.npz')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 轨迹
        ax1.plot(df['time'], df['x'], 'b-', lw=1)
        ax1.set_ylabel('x',fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        ax1.tick_params(axis='y', labelsize=20) 
        # ax1.set_title('Trajectory')
        ax1.grid(True)
        
        # 误差 E(t)
        ax2.plot(err_data['t_err'], err_data['err'], 'ko', alpha=0.8, lw=0.5, markersize=1, label='Instant Error $E(t)$')
        ax2.plot(err_data['t_ma'], err_data['ma'], 'r-', lw=2, label='Periodicity Error $\overline{E(t)}$')
        ax2.set_ylabel('$\|x(t)-x(t-T)\|$',fontsize=20)
        ax2.set_xlabel('Time',fontsize=20)
        ax2.set_yscale('linear') 
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # ax2.set_title('Convergence to Steady State')
        ax2.legend(fontsize=15)
        ax2.grid(True)

    def plot_task2(self):
        print("Plotting Task 2...")
        param_names = ['delta', 'alpha', 'beta', 'gamma', 'omega']
        pairs = list(itertools.combinations(param_names, 2))
        
        # fig, axes = plt.subplots(3, 4, figsize=(16, 9))
        # axes = axes.flatten()
        # fig.suptitle('Steady State Time Phase Diagrams')
        fig = plt.figure(figsize=(15, 9))
        gs = GridSpec(3, 4, hspace=0.4, wspace=0.4) #行列间距

        axes = []
        for row in range(2):
            for col in range(4):
                axes.append(fig.add_subplot(gs[row, col]))

        axes.append(fig.add_subplot(gs[2, 1]))
        axes.append(fig.add_subplot(gs[2, 2]))

        # fig.suptitle('Steady State Time Phase Diagrams',fontsize=25, fontweight='bold')
        
        for idx, (p1, p2) in enumerate(pairs):
            file_path = f'Data/task2_scan_{p1}_{p2}.csv'
            if not os.path.exists(file_path): 
                axes[idx].set_visible(False)
                continue
            
            df = pd.read_csv(file_path)
            
            # 转换为矩阵格式用于绘图
            pivot = df.pivot(index=p2, columns=p1, values='steady_time')
            X = pivot.columns.values
            Y = pivot.index.values
            Z = pivot.values
            
            ax = axes[idx]
            cp = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
            fig.colorbar(cp, ax=ax)
            ax.set_xlabel(p1, fontsize='large', fontweight='bold')
            ax.set_ylabel(p2, fontsize='large', fontweight='bold')
            ax.set_title(f'{p1} vs {p2}')
            
        plt.tight_layout()

    def plot_task3(self):
        print("Plotting Task 3...")
        df_fwd = pd.read_csv('Data/task3_fwd.csv')
        df_bwd = pd.read_csv('Data/task3_bwd.csv')
        # df_trajectory = pd.read_csv('Data/task3_trajectory_scan_frequency.csv')

        # plt.figure(figsize=(15,6))
        # plt.plot(df_trajectory['time'], df_trajectory['x'], '-.', markersize=0.5, label='Trajectory Over Time')
        # # plt.xlabel('time', fontsize='large', fontweight='bold')
        # # plt.ylabel('x', fontsize='large', fontweight = 'bold')
        # plt.legend()
        # plt.grid(True)

        plt.figure(figsize=(8, 6))
        plt.plot(df_fwd['omega'], df_fwd['amp'], 'b-o', markersize=4, label='Forward Sweep')
        plt.plot(df_bwd['omega'], df_bwd['amp'], 'r--*', markersize=4, label='Backward Sweep')
        plt.title('Hysteresis Loop',fontsize='large')
        plt.xlabel('Frequency $\omega$', fontsize='large')
        plt.ylabel('Amplitude', fontsize='large')
        plt.legend()
        plt.grid(True)

    def plot_task32(self):
        print("Plotting Task 3 Detailed...")
        b = [0, 0.01, 0.1, 0.3, 0.5, 1, 2, 4, 10]
        x = np.linspace(0.5, 5.5, 60); y = 2.5 /np.sqrt((1 - x*x)*(1 - x*x) + 0.04*x*x)
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'r-.', markersize=8, linewidth=3, label=rf'Theoretical Curve with $\beta=0$')
        for i in b:
            df_fwd = pd.read_csv(f'Data/b={i}/task3_fwd.csv')
            df_bwd = pd.read_csv(f'Data/b={i}/task3_bwd.csv')
            plt.plot(df_fwd['omega'], df_fwd['amp'], '-o', markersize=8, linewidth=1 ,label=rf'Forward Sweep $\beta=${i}')
            plt.plot(df_bwd['omega'], df_bwd['amp'], '--*', markersize=8, linewidth=1 ,label=rf'Backward Sweep $\beta=${i}')

        # plt.title('Hysteresis Loop',fontsize=20)
        plt.xlabel('Frequency $\omega$', fontsize=20)
        plt.ylabel('Amplitude', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=15,ncol=2)
        plt.grid(True)

    def plot_task4(self):
        print("Plotting Task 4...")
        df_traj = pd.read_csv('Data/task4_traj.csv')
        df_poincare = pd.read_csv('Data/task4_poincare.csv')
        
        fig = plt.figure(figsize=(12, 5))
        
        # 相图
        ax1 = fig.add_subplot(121)
        # 只画最后一部分看吸引子
        steady_traj = df_traj.iloc[-5000:] 
        ax1.plot(steady_traj['x'], steady_traj['v'], 'k-', lw=0.3, alpha=0.6)
        ax1.set_title('Phase Portrait (Chaos)')
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$v$')
        
        # 庞加莱截面
        ax2 = fig.add_subplot(122)
        ax2.scatter(df_poincare['x'], df_poincare['v'], s=2, c='r', marker='.')
        ax2.set_title('Poincaré Section')
        ax2.set_xlabel('$x$')
        ax2.set_ylabel('$v$')
        
        plt.tight_layout()

def main():
    # 1. 计算阶段
    computer = ComputationManager()
    computer.run_task1_trajectory()
    # computer.run_task2_parameter_scan()
    # computer.run_task3_hysteresis()
    # computer.run_task4_chaos()
    
    print("\nAll computations finished. Starting visualization...\n")
    
    # 2. 绘图阶段
    plotter = VisualizationManager()
    # plt.rcParams.update({
    #     "text.usetex": True,        # 启用LaTeX渲染
    #     "font.family": "serif",     # -serif字体更兼容LaTeX
    #     "font.serif": ["Times New Roman"],  # 指定支持希腊字母的字体
    #     "axes.unicode_minus": False # 避免负号显示异常（可选）
    # })
    plotter.plot_task1()
    # plotter.plot_task2()
    # plotter.plot_task3()
    # plotter.plot_task32()
    # plotter.plot_task4()
    
    # 3. 统一展示
    print("Displaying all plots...")
    plt.show()

if __name__ == "__main__":
    main()
