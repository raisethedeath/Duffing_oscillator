import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 1. 定义达芬振子的微分方程
def duffing_deriv(state, t, delta, alpha, beta, gamma, omega):
    x, v = state  # state = [位移, 速度]
    # dxdt = v
    # dvdt = -delta*v - alpha*x - beta*x^3 + gamma*cos(omega*t)
    dxdt = v
    dvdt = -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
    return [dxdt, dvdt]

# 2. 实验参数设置
delta = 0.1    # 阻尼
alpha = 1.0    # 线性刚度
beta = 0.0     # 非线性刚度 (硬弹簧)
gamma = 2.5    # 驱动力幅值 (需要足够大以产生多稳态)

# 频率扫描范围
w_start = 0.5
w_end = 10.5
steps = 100
w_forward = np.linspace(w_start, w_end, steps)
w_backward = np.linspace(w_end, w_start, steps)

# 模拟参数
t_max = 200       # 每次模拟的总时间
dt = 0.05         # 时间步长
t_eval = np.arange(0, t_max, dt)
transient_cut = int(len(t_eval) * 0.6) # 舍弃前60%的数据作为瞬态

# 3. 数值实验函数
def run_sweep(omega_array, initial_state):
    amplitudes = []
    current_state = initial_state
    
    for w in omega_array:
        # 求解ODE
        sol = odeint(duffing_deriv, current_state, t_eval, args=(delta, alpha, beta, gamma, w))
        
        # 提取稳态部分 (去除瞬态)
        x_steady = sol[transient_cut:, 0]
        
        # 计算振幅 (取稳态部分的最大绝对值)
        amp = (np.max(x_steady) - np.min(x_steady)) / 2
        amplitudes.append(amp)
        
        # 关键：更新初始状态为本次模拟的终点 (绝热延拓)
        current_state = sol[-1]
        
    return np.array(amplitudes)

# 4. 执行实验
print("正在进行正向频率扫描...")
# 初始静止
state0 = [0.0, 0.0] 
amp_fwd = run_sweep(w_forward, state0)

print("正在进行反向频率扫描...")
# 使用正向扫描的最后状态作为反向的开始
state_end_fwd = odeint(duffing_deriv, state0, t_eval, args=(delta, alpha, beta, gamma, w_forward[-1]))[-1]
amp_bwd = run_sweep(w_backward, state_end_fwd)

# 5. 结果可视化
plt.figure(figsize=(10, 6))
plt.plot(w_forward, amp_fwd, 'b-o', label='Forward Sweep (Frequency Increasing)', markersize=4)
plt.plot(w_backward, amp_bwd, 'r-o', label='Backward Sweep (Frequency Decreasing)', markersize=4)

plt.title(f'Hysteresis Loop in Duffing Oscillator\nParameters: $\delta={delta}, \\alpha={alpha}, \\beta={beta}, \gamma={gamma}$')
plt.xlabel('Driving Frequency $\omega$')
plt.ylabel('Steady State Amplitude')
plt.legend()
plt.grid(True)

# # 标注跳跃点
# plt.annotate('Jump Down', xy=(2.5, 1.5), xytext=(3.0, 2.5),
#              arrowprops=dict(facecolor='black', shrink=0.05))
# plt.annotate('Jump Up', xy=(1.5, 1.0), xytext=(0.8, 2.0),
#              arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()
